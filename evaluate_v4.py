"""
evaluate_v4.py

Evaluation harness for the Cursor-style (pipeline_v4) SVA assertion pipeline.
Runs JasperGold prove on all generated sva_assertion.sv files, computes a richer
set of metrics than the original evaluate.py, and optionally checks vacuity.

Metrics per spec:
  syntax        - 1.0 if no compilation/syntax error
  functionality - proven / total
  func_relaxed  - (proven + undetermined) / total
  proven        - count of proven properties
  falsified     - count of falsified/cex properties
  undetermined  - count of undetermined properties
  total         - total properties
  vacuity       - "vacuous" | "non_vacuous" | "not_checked" | "error"
  vacuity_details - per-property vacuity string from JG
  jg_iterations - from agent_result.json if available, else 0
  tool_calls    - from agent_result.json if available, else 0
  wall_time_s   - from agent_result.json if available, else 0.0
  status        - "evaluated" | "skipped" | "error"

Usage:
  python evaluate_v4.py --debug_dir results/v4_pipeline --designs_csv big_designs.csv
                        [--output results_eval.csv] [--vacuity] [--parallel 4]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# JasperGold SSH helper (mirrors tools.py pattern)
# ---------------------------------------------------------------------------

# JasperGold SSH target and remote scratch directory are environment-specific;
# configure a host alias in ~/.ssh/config (recommended, key-based auth) rather
# than hardcoding credentials here. See README.md "Setup".
JG_SSH_HOST = os.environ.get("JASPERGOLD_SSH_HOST", "jaspergold")
JG_REMOTE_DIR = os.environ.get("JASPERGOLD_REMOTE_DIR", "~/proofloop_jg_work")


def _ssh_base_cmd():
    """Build the SSH invocation used for all JasperGold calls (see tools.py)."""
    cmd = []
    if os.environ.get("SSHPASS"):
        cmd += ['sshpass', '-e']
    cmd += [
        'ssh',
        '-o', 'ConnectTimeout=15',
        '-o', 'ServerAliveInterval=10',
        '-o', 'ServerAliveCountMax=3',
        '-o', 'StrictHostKeyChecking=no',
        JG_SSH_HOST, 'bash',
    ]
    return cmd


def _run_jg_ssh(module_name: str, rtl_code: str, tcl_body: str,
                tag: str, timeout: int = 300) -> str:
    """Run a JasperGold TCL script on the configured JasperGold host via SSH.

    Creates a temp dir on the remote, writes dut_with_sva.sv and run.tcl,
    runs JasperGold batch, then cleans up.

    Returns raw stdout from JasperGold (comment lines filtered).
    """
    remote_dir = f"{JG_REMOTE_DIR}/tmp_jg_eval_{module_name}_{tag}_{os.getpid()}"

    bash_script = f'''
mkdir -p {remote_dir}
cd {remote_dir}

cat > dut_with_sva.sv << 'PYEOF'
{rtl_code}
PYEOF

cat > run.tcl << 'PYEOF'
{tcl_body}
PYEOF

jg -allow_unsupported_OS -batch run.tcl 2>&1 | grep -v "^#"

cd {JG_REMOTE_DIR}
rm -rf {remote_dir}
'''

    ssh_cmd = _ssh_base_cmd()

    result = subprocess.run(
        ssh_cmd,
        input=bash_script,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    raw = (result.stdout + "\n" + result.stderr).strip()
    lines = raw.splitlines()
    prompt_start = next(
        (i for i, ln in enumerate(lines) if not ln.startswith('%')), 0
    )
    return "\n".join(lines[prompt_start:])


# ---------------------------------------------------------------------------
# SVA helpers
# ---------------------------------------------------------------------------

def _extract_sva(text: str) -> str:
    """Strip markdown fenced code blocks if present, returning bare SVA."""
    pattern = re.compile(r"```(?:systemverilog|sv|verilog)?\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _inject_sva(rtl_code: str, sva_code: str, top_module: str = "") -> str:
    """Inject SVA text before the endmodule of the top-level module."""
    injection = "\n// --- Injected SVA assertions ---\n" + sva_code + "\n\n"

    if top_module:
        mod_pattern = re.compile(
            r'\bmodule\s+' + re.escape(top_module) + r'\b', re.MULTILINE
        )
        mod_match = mod_pattern.search(rtl_code)
        if mod_match:
            search_start = mod_match.end()
            depth = 1
            pos = search_start
            while pos < len(rtl_code):
                next_mod = re.search(r'\bmodule\s+\w+', rtl_code[pos:])
                next_end = re.search(r'\bendmodule\b', rtl_code[pos:])
                if next_end is None:
                    break
                end_abs = pos + next_end.start()
                if next_mod and (pos + next_mod.start()) < end_abs:
                    depth += 1
                    pos = pos + next_mod.end()
                    continue
                depth -= 1
                if depth == 0:
                    return rtl_code[:end_abs] + injection + rtl_code[end_abs:]
                pos = end_abs + len("endmodule")

    # Fallback: inject before the LAST endmodule
    pos = rtl_code.rfind("endmodule")
    if pos != -1:
        return rtl_code[:pos] + injection + rtl_code[pos:]
    return rtl_code + "\n" + sva_code


def _extract_defines(rtl_code: str) -> str:
    """Extract `define values from RTL and build JasperGold +define+ string."""
    defaults = {"WIDTH": "128", "DEPTH": "8", "NS": "8", "OPD": "2"}
    found = {}
    for m in re.finditer(r"^\s*`define\s+(\w+)\s+(\d+)", rtl_code, re.MULTILINE):
        name, value = m.group(1), m.group(2)
        if name in defaults:
            found[name] = value
    merged = {**defaults, **found}
    kv_parts = "+".join(f"{k}={v}" for k, v in sorted(merged.items()))
    return f"+define+{kv_parts}"


def _detect_clock_reset(rtl_code: str) -> dict:
    """Detect clock and reset signal names from RTL port declarations."""
    clock = "clk"
    reset_name = ""
    reset_active_low = True

    for pat, name in [
        (r'\binput\b[^;]*\bclk\b', 'clk'),
        (r'\binput\b[^;]*\bCLK\b', 'CLK'),
        (r'\binput\b[^;]*\bclock\b', 'clock'),
    ]:
        if re.search(pat, rtl_code):
            clock = name
            break

    for pat, active_low in [
        (r'\binput\b[^;]*\b(reset_)\b', True),
        (r'\binput\b[^;]*\b(rst_n)\b', True),
        (r'\binput\b[^;]*\b(reset_n)\b', True),
        (r'\binput\b[^;]*\b(resetn)\b', True),
        (r'\binput\b[^;]*\b(rst)\b', False),
        (r'\binput\b[^;]*\b(reset)\b', False),
    ]:
        m = re.search(pat, rtl_code)
        if m:
            reset_name = m.group(1)
            reset_active_low = active_low
            break

    tcl_lines = f'catch {{clock {clock}}}\n'
    if reset_name:
        if reset_active_low:
            tcl_lines += f'catch {{reset -expression {{!{reset_name}}}}}\n'
        else:
            tcl_lines += f'catch {{reset {reset_name}}}\n'
    else:
        tcl_lines += 'catch {reset -expression {!reset_}}\n'
        tcl_lines += 'catch {reset rst}\n'

    return {"clock": clock, "reset_name": reset_name,
            "reset_active_low": reset_active_low, "reset_tcl": tcl_lines}


# ---------------------------------------------------------------------------
# Metric parsers
# ---------------------------------------------------------------------------

def _parse_prove_output(raw: str) -> Dict:
    """Parse JG prove output → syntax/proven/falsified/undetermined."""
    metrics = {
        "syntax": 1.0,
        "proven": 0,
        "falsified": 0,
        "undetermined": 0,
        "total": 0,
        "errors": [],
    }

    if re.search(r"syntax error", raw, re.IGNORECASE):
        metrics["syntax"] = 0.0
        metrics["errors"] = re.findall(
            r"(?:ERROR|syntax error)[^\n]*", raw, re.IGNORECASE
        )[:10]
        return metrics

    all_errors = re.findall(r"\[?ERROR[^\n]*", raw, re.IGNORECASE)
    if all_errors and not re.search(r"===EVAL_RESULTS===", raw):
        metrics["syntax"] = 0.0
        metrics["errors"] = all_errors[:10]
        return metrics

    proven      = len(re.findall(r"STATUS:\s*proven",            raw, re.IGNORECASE))
    falsified   = len(re.findall(r"STATUS:\s*(?:falsified|cex)", raw, re.IGNORECASE))
    undetermined= len(re.findall(r"STATUS:\s*undetermined",      raw, re.IGNORECASE))

    # Fallback: scan summary lines like "proofs: proven undetermined"
    if proven + falsified + undetermined == 0:
        for proof_line in re.findall(r"proofs?:[^\n]*", raw, re.IGNORECASE):
            for token in proof_line.split(":")[-1].strip().split():
                tok = token.lower().strip()
                if tok == "proven":
                    proven += 1
                elif tok in ("falsified", "cex"):
                    falsified += 1
                elif tok == "undetermined":
                    undetermined += 1

    metrics["proven"]       = proven
    metrics["falsified"]    = falsified
    metrics["undetermined"] = undetermined
    metrics["total"]        = proven + falsified + undetermined
    return metrics


def _parse_vacuity_output(raw: str) -> Dict[str, str]:
    """Parse JG check_vacuity output → {status, details}.

    Returns:
        status:  "vacuous" | "non_vacuous" | "error" | "not_checked"
        details: raw vacuity section text
    """
    vac_match = re.search(
        r"===VACUITY_START===(.*?)===VACUITY_END===",
        raw,
        re.DOTALL,
    )
    if not vac_match:
        return {"status": "error", "details": "No vacuity section in JG output."}

    vac_raw = vac_match.group(1).strip()
    if not vac_raw:
        return {"status": "error", "details": "Empty vacuity section."}

    # Vacuous if any line contains "vacuous" (case-insensitive)
    if re.search(r"\bvacuous\b", vac_raw, re.IGNORECASE):
        return {"status": "vacuous", "details": vac_raw}
    return {"status": "non_vacuous", "details": vac_raw}


# ---------------------------------------------------------------------------
# RTL collection helpers
# ---------------------------------------------------------------------------

def _collect_module_rtls(design_dir: str) -> str:
    """Concatenate all per-module rtl.sv files found under design_dir."""
    parts: List[str] = []
    try:
        for entry in sorted(os.scandir(design_dir), key=lambda e: e.name):
            if not entry.is_dir() or entry.name == "specs":
                continue
            rtl_path = os.path.join(entry.path, "rtl.sv")
            if os.path.exists(rtl_path):
                with open(rtl_path) as f:
                    parts.append(f.read())
    except OSError:
        pass
    return "\n\n".join(parts)


def _find_top_module(design_dir: str, rtl_code: str = "") -> str:
    """Determine the top module name from design_graph.json, dir listing, or RTL."""
    graph_path = os.path.join(design_dir, "design_graph.json")
    if os.path.exists(graph_path):
        try:
            with open(graph_path) as f:
                graph = json.load(f)
            modules = graph.get("sorted_modules", [])
            if modules:
                return modules[0]
        except Exception:
            pass

    # Fall back: first subdir of design_dir (excluding 'specs')
    try:
        module_dirs = sorted(
            e.name for e in os.scandir(design_dir)
            if e.is_dir() and e.name != "specs"
        )
        if module_dirs:
            return module_dirs[0]
    except OSError:
        pass

    # Last resort: parse first module declaration from RTL
    if rtl_code:
        m = re.search(r"\bmodule\s+(\w+)", rtl_code)
        if m:
            return m.group(1)

    return ""


# ---------------------------------------------------------------------------
# Agent result reader
# ---------------------------------------------------------------------------

def _read_agent_result(spec_dir: str) -> Dict:
    """Read agent_result.json if present, else return zero-value defaults."""
    defaults = {"jg_iterations": 0, "tool_calls": 0, "wall_time_s": 0.0}
    result_path = os.path.join(spec_dir, "agent_result.json")
    if not os.path.exists(result_path):
        return defaults
    try:
        with open(result_path) as f:
            data = json.load(f)
        return {
            "jg_iterations": int(data.get("jg_iterations", 0)),
            "tool_calls":    int(data.get("tool_calls", 0)),
            "wall_time_s":   float(data.get("wall_time_s", 0.0)),
        }
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_spec(
    design_id: str,
    spec_id: str,
    sva_path: str,
    designs_df: Optional[pd.DataFrame],
    debug_dir: str,
    do_vacuity: bool = False,
) -> Dict:
    """Evaluate a single SVA assertion file with JasperGold.

    Steps:
      1. Read SVA code (strip markdown fences if needed).
      2. Collect module RTLs from design_dir subdirs; fall back to designs_df RTL.
      3. Inject SVA before first endmodule of concatenated RTL.
      4. Run JG prove, parse syntax/proven/falsified/undetermined.
      5. If do_vacuity and syntax=1.0 and total>0: run vacuity check.
      6. Read agent_result.json for pipeline telemetry.
      7. Write eval_jg_output.txt, eval_metrics.json, eval_vacuity.txt.
      8. Return metrics dict.

    Returns a metrics dict conforming to the evaluate_v4 schema.
    """
    spec_dir   = os.path.dirname(sva_path)
    design_dir = os.path.join(debug_dir, design_id)

    base_row = {
        "design_id":       design_id,
        "spec_id":         spec_id,
        "syntax":          0.0,
        "functionality":   0.0,
        "func_relaxed":    0.0,
        "proven":          0,
        "falsified":       0,
        "undetermined":    0,
        "total":           0,
        "vacuity":         "not_checked",
        "vacuity_details": "",
        "jg_iterations":   0,
        "tool_calls":      0,
        "wall_time_s":     0.0,
        "status":          "error",
    }

    # -- Read agent telemetry --
    agent_info = _read_agent_result(spec_dir)
    base_row.update(agent_info)

    # -- Read SVA --
    try:
        with open(sva_path) as f:
            sva_raw = f.read()
    except OSError as e:
        base_row["status"] = "error"
        return base_row

    sva_code = _extract_sva(sva_raw)
    if not sva_code or sva_code.startswith("// SVA generation failed"):
        base_row["status"] = "skipped"
        return base_row

    # -- Collect RTL --
    # Get original RTL from designs_df (has `define lines) for define extraction
    original_rtl = ""
    if designs_df is not None and design_id in designs_df.index:
        original_rtl = str(designs_df.loc[design_id, "rtl"])

    rtl_code = _collect_module_rtls(design_dir)
    if not rtl_code:
        flat = os.path.join(design_dir, "rtl.sv")
        if os.path.exists(flat):
            with open(flat) as f:
                rtl_code = f.read()
        elif original_rtl:
            rtl_code = original_rtl
        else:
            base_row["status"] = "error"
            return base_row

    # Use original RTL or defines.sv for define extraction (per-module files lack `define lines)
    defines_path = os.path.join(design_dir, "defines.sv")
    if os.path.exists(defines_path):
        with open(defines_path) as f:
            define_source = f.read()
    elif original_rtl:
        define_source = original_rtl
    else:
        define_source = rtl_code

    top_module = _find_top_module(design_dir, rtl_code)
    if not top_module:
        base_row["status"] = "error"
        return base_row

    # -- Inject SVA --
    combined_rtl = _inject_sva(rtl_code, sva_code, top_module=top_module)

    # -- Build prove TCL --
    cr = _detect_clock_reset(rtl_code)
    prove_tcl = (
        f"analyze -sv {{{_extract_defines(define_source)}}} dut_with_sva.sv\n"
        f"elaborate -top {top_module}\n"
        f"{cr['reset_tcl']}"
        f"prove -all -time_limit 1m\n"
        f"puts \"===EVAL_RESULTS===\"\n"
        f"foreach p [get_property_list] {{\n"
        f"    set st [get_status $p]\n"
        f"    puts \"PROP: $p STATUS: $st\"\n"
        f"}}\n"
        f"puts \"===EVAL_END===\"\n"
        f"exit\n"
    )

    # -- Run prove --
    try:
        tag      = re.sub(r"[^A-Za-z0-9_]", "_", spec_id)[:30]
        jg_out   = _run_jg_ssh(top_module, combined_rtl, prove_tcl,
                               tag=tag, timeout=300)
    except subprocess.TimeoutExpired:
        jg_out = "ERROR: SSH timeout (300s)"
    except Exception as e:
        jg_out = f"ERROR: {e}"

    # Write raw JG output
    try:
        with open(os.path.join(spec_dir, "eval_jg_output.txt"), "w") as f:
            f.write(jg_out)
    except OSError:
        pass

    # -- Parse prove metrics --
    parsed = _parse_prove_output(jg_out)
    total  = parsed["total"]

    base_row["syntax"]      = parsed["syntax"]
    base_row["proven"]      = parsed["proven"]
    base_row["falsified"]   = parsed["falsified"]
    base_row["undetermined"]= parsed["undetermined"]
    base_row["total"]       = total

    if total > 0:
        base_row["functionality"] = parsed["proven"] / total
        base_row["func_relaxed"]  = (parsed["proven"] + parsed["undetermined"]) / total
    else:
        base_row["functionality"] = 0.0
        base_row["func_relaxed"]  = 0.0

    base_row["status"] = "evaluated"

    # -- Vacuity check --
    if do_vacuity and parsed["syntax"] == 1.0 and total > 0:
        vacuity_tcl = (
            f"analyze -sv {{{_extract_defines(define_source)}}} dut_with_sva.sv\n"
            f"elaborate -top {top_module}\n"
            f"{cr['reset_tcl']}"
            f"prove -all -time_limit 1m\n"
            f"puts \"===VACUITY_START===\"\n"
            f"catch {{check_vacuity -all}}\n"
            f"puts \"===VACUITY_END===\"\n"
            f"exit\n"
        )
        try:
            vac_tag = f"{tag}_vac"
            vac_out = _run_jg_ssh(top_module, combined_rtl, vacuity_tcl,
                                  tag=vac_tag, timeout=300)
            vac_result = _parse_vacuity_output(vac_out)

            try:
                with open(os.path.join(spec_dir, "eval_vacuity.txt"), "w") as f:
                    f.write(vac_out)
            except OSError:
                pass

            base_row["vacuity"]         = vac_result["status"]
            base_row["vacuity_details"] = vac_result["details"]

        except subprocess.TimeoutExpired:
            base_row["vacuity"] = "error"
            base_row["vacuity_details"] = "SSH timeout during vacuity check."
        except Exception as e:
            base_row["vacuity"] = "error"
            base_row["vacuity_details"] = str(e)
    else:
        base_row["vacuity"] = "not_checked"

    # -- Write per-spec metrics JSON --
    metrics_out = {k: v for k, v in base_row.items()
                   if k not in ("vacuity_details",)}
    metrics_out["vacuity_details"] = base_row["vacuity_details"]
    try:
        with open(os.path.join(spec_dir, "eval_metrics.json"), "w") as f:
            json.dump(base_row, f, indent=2)
    except OSError:
        pass

    return base_row


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def collect_spec_tasks(debug_dir: str, designs_df: Optional[pd.DataFrame]) -> List[Dict]:
    """Walk debug_dir and collect all (design_id, spec_id, sva_path) triples."""
    tasks = []
    try:
        design_dirs = sorted(
            e.name for e in os.scandir(debug_dir)
            if e.is_dir() and e.name not in ("specs",)
        )
    except OSError:
        return tasks

    for design_id in design_dirs:
        design_dir = os.path.join(debug_dir, design_id)
        specs_root = os.path.join(design_dir, "specs")
        if not os.path.isdir(specs_root):
            continue
        try:
            spec_ids = sorted(
                e.name for e in os.scandir(specs_root)
                if e.is_dir()
            )
        except OSError:
            continue

        for spec_id in spec_ids:
            spec_dir = os.path.join(specs_root, spec_id)
            sva_path = os.path.join(spec_dir, "sva_assertion.sv")
            if not os.path.exists(sva_path):
                # Fallback: double-nested path from pre-fix runs
                sva_path = os.path.join(spec_dir, spec_id, "sva_assertion.sv")
                if not os.path.exists(sva_path):
                    continue
            tasks.append({
                "design_id": design_id,
                "spec_id":   spec_id,
                "sva_path":  sva_path,
            })
    return tasks


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def evaluate_debug_dir(
    debug_dir: str,
    designs_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    do_vacuity: bool = False,
    parallel: int = 1,
) -> List[Dict]:
    """Evaluate all SVA assertions in debug_dir.

    Args:
        debug_dir:   Root debug output directory.
        designs_csv: Path to designs.csv (fallback RTL source).
        output_csv:  Output CSV path.
        do_vacuity:  Run vacuity check per spec.
        parallel:    Number of concurrent threads.

    Returns:
        List of per-spec metric dicts.
    """
    designs_df = None
    if designs_csv and os.path.exists(designs_csv):
        df = pd.read_csv(designs_csv)
        designs_df = df.set_index("id")

    tasks = collect_spec_tasks(debug_dir, designs_df)
    print(f"Found {len(tasks)} specs to evaluate in {debug_dir}")

    # -- Resume: skip already-evaluated specs --
    pending = []
    skipped_resume = 0
    for task in tasks:
        metrics_path = os.path.join(
            debug_dir, task["design_id"], "specs", task["spec_id"], "eval_metrics.json"
        )
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    cached = json.load(f)
                pending.append(("cached", cached))
                skipped_resume += 1
                continue
            except Exception:
                pass
        pending.append(("run", task))

    print(f"  Resuming: {skipped_resume} already evaluated, "
          f"{len(pending) - skipped_resume} remaining.")

    results: List[Dict] = []

    def _run_task(item):
        kind, data = item
        if kind == "cached":
            return data
        design_id = data["design_id"]
        spec_id   = data["spec_id"]
        sva_path  = data["sva_path"]
        print(f"  [{design_id}] Evaluating {spec_id} ...")
        row = evaluate_spec(
            design_id, spec_id, sva_path,
            designs_df, debug_dir, do_vacuity=do_vacuity,
        )
        _print_row(row)
        return row

    if parallel <= 1:
        for item in pending:
            results.append(_run_task(item))
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(_run_task, item): item for item in pending}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"  [ERROR] Task failed: {e}")

    return results


def _print_row(row: Dict):
    """Print a compact one-line summary for a spec result."""
    print(
        f"    [{row['design_id']}] {row['spec_id']}: "
        f"syntax={row['syntax']:.1f} "
        f"func={row['functionality']:.2f} "
        f"relaxed={row['func_relaxed']:.2f} "
        f"(proven={row['proven']}, falsified={row['falsified']}, "
        f"undetermined={row['undetermined']}) "
        f"vacuity={row['vacuity']}"
    )


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(results: List[Dict]) -> Dict:
    """Aggregate per-spec metrics into an evaluation summary dict."""
    evaluated   = [r for r in results if r["status"] == "evaluated"]
    skipped_cnt = sum(1 for r in results if r["status"] == "skipped")
    error_cnt   = sum(1 for r in results if r["status"] == "error")
    n = len(evaluated)

    def safe_avg(key):
        return sum(r[key] for r in evaluated) / n if n else 0.0

    vacuous_count     = sum(1 for r in evaluated if r["vacuity"] == "vacuous")
    non_vacuous_count = sum(1 for r in evaluated if r["vacuity"] == "non_vacuous")

    total_proven      = sum(r["proven"]      for r in evaluated)
    total_falsified   = sum(r["falsified"]   for r in evaluated)
    total_undetermined= sum(r["undetermined"]for r in evaluated)

    return {
        "total_specs":         len(results),
        "evaluated":           n,
        "skipped":             skipped_cnt,
        "errors":              error_cnt,
        "avg_syntax":          round(safe_avg("syntax"),        4),
        "avg_functionality":   round(safe_avg("functionality"), 4),
        "avg_func_relaxed":    round(safe_avg("func_relaxed"),  4),
        "total_proven":        total_proven,
        "total_falsified":     total_falsified,
        "total_undetermined":  total_undetermined,
        "vacuous_count":       vacuous_count,
        "non_vacuous_count":   non_vacuous_count,
        "avg_tool_calls":      round(safe_avg("tool_calls"),    2),
        "avg_jg_iterations":   round(safe_avg("jg_iterations"), 2),
        "avg_wall_time_s":     round(safe_avg("wall_time_s"),   2),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Cursor-style (v4) SVA assertions with JasperGold. "
            "Walks debug_dir for sva_assertion.sv files, runs JG prove, "
            "and computes syntax/functionality/vacuity metrics."
        )
    )
    parser.add_argument(
        "--debug_dir", required=True,
        help="Root debug output directory (e.g. results/v4_pipeline).",
    )
    parser.add_argument(
        "--designs_csv", default=None,
        help="Path to designs.csv — used as fallback RTL source.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: <debug_dir>/evaluation_results.csv).",
    )
    parser.add_argument(
        "--vacuity", action="store_true", default=False,
        help="Enable vacuity checking per spec (~30s extra per spec).",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of concurrent spec evaluations (default: 1).",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.debug_dir, "evaluation_results.csv")

    print(f"evaluate_v4.py")
    print(f"  debug_dir:   {args.debug_dir}")
    print(f"  designs_csv: {args.designs_csv}")
    print(f"  output:      {output_path}")
    print(f"  vacuity:     {args.vacuity}")
    print(f"  parallel:    {args.parallel}")
    print()

    results = evaluate_debug_dir(
        debug_dir=args.debug_dir,
        designs_csv=args.designs_csv,
        output_csv=output_path,
        do_vacuity=args.vacuity,
        parallel=args.parallel,
    )

    if not results:
        print("No assertions found to evaluate.")
        return

    # -- Write CSV --
    fieldnames = [
        "design_id", "spec_id",
        "syntax", "functionality", "func_relaxed",
        "proven", "falsified", "undetermined", "total",
        "vacuity", "vacuity_details",
        "jg_iterations", "tool_calls", "wall_time_s",
        "status",
    ]
    # Ensure all rows have all fields
    for row in results:
        for fn in fieldnames:
            row.setdefault(fn, "")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")

    # -- Compute and print summary --
    summary = compute_summary(results)

    summary_path = os.path.join(args.debug_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    n = summary["evaluated"]
    print(f"\n{'='*65}")
    print(f"EVALUATION SUMMARY  ({n} evaluated / {summary['total_specs']} total)")
    print(f"{'='*65}")
    print(f"  avg_syntax:         {summary['avg_syntax']:.4f}")
    print(f"  avg_functionality:  {summary['avg_functionality']:.4f}")
    print(f"  avg_func_relaxed:   {summary['avg_func_relaxed']:.4f}")
    print(f"  total_proven:       {summary['total_proven']}")
    print(f"  total_falsified:    {summary['total_falsified']}")
    print(f"  total_undetermined: {summary['total_undetermined']}")
    if args.vacuity:
        print(f"  vacuous_count:      {summary['vacuous_count']}")
        print(f"  non_vacuous_count:  {summary['non_vacuous_count']}")
    print(f"  avg_tool_calls:     {summary['avg_tool_calls']:.2f}")
    print(f"  avg_jg_iterations:  {summary['avg_jg_iterations']:.2f}")
    print(f"  avg_wall_time_s:    {summary['avg_wall_time_s']:.2f}")
    print(f"  skipped:            {summary['skipped']}")
    print(f"  errors:             {summary['errors']}")
    print(f"{'='*65}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
