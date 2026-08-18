"""
evaluate_ab.py — Evaluate generated SVA assertions against JasperGold.

Walks a results directory, finds sva_assertion.sv files, runs JG prove on each,
and computes Pass/CEX/Error metrics matching AssertionBench paper methodology.

Metrics per design:
  - total: number of assert property statements
  - pass (proven): JG proves the property
  - cex (falsified): JG finds a counterexample
  - error: syntax/compilation error
  - pass_rate = pass / total
"""

try:
    import pysqlite3, sys as _sys_sq; _sys_sq.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CURSOR_DIR = os.path.dirname(_THIS_DIR)
if _CURSOR_DIR not in sys.path:
    sys.path.insert(0, _CURSOR_DIR)

from loader import load_test_designs


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(*args, **kwargs):
    print(f"[{_ts()}]", *args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# JG SSH (reuse from tools.py)
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


def _run_jg_ssh(module_name, rtl_code, tcl_body, tag="eval", timeout=300):
    """Run JG prove via SSH to the configured JasperGold host."""
    remote_dir = f"{JG_REMOTE_DIR}/tmp_jg_{module_name}_{tag}_{os.getpid()}"

    bash_script = f'''
mkdir -p {remote_dir}
cd {remote_dir}

cat > dut.sv << 'PYEOF'
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

    proc = subprocess.Popen(
        ssh_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = proc.communicate(input=bash_script, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        raise

    raw = (stdout + "\n" + stderr).strip()
    lines = raw.splitlines()
    start = next((i for i, ln in enumerate(lines) if not ln.startswith('%')), 0)
    return "\n".join(lines[start:])


def _parse_prove_results(raw):
    """Parse JG prove output — same logic as tools.py."""
    result = {
        "status": "unknown",
        "proven": 0,
        "falsified": 0,
        "undetermined": 0,
        "properties": [],
        "errors": [],
        "raw": raw,
    }

    if re.search(r"syntax error", raw, re.IGNORECASE):
        result["status"] = "syntax_error"
        result["errors"] = re.findall(r"(?:ERROR|syntax error)[^\n]*", raw, re.IGNORECASE)[:10]
        return result

    all_errors = re.findall(r"\[?ERROR[^\n]*", raw, re.IGNORECASE)
    if all_errors and not re.search(r"===RESULTS===", raw):
        result["status"] = "compilation_error"
        result["errors"] = all_errors[:10]
        return result

    results_section = ""
    m = re.search(r"===RESULTS===(.*?)(?:===END===|$)", raw, re.DOTALL)
    if m:
        results_section = m.group(1)

    prop_lines = re.findall(r"PROP:\s*(\S+)\s+STATUS:\s*(\S+)", results_section, re.IGNORECASE)
    for prop_name, prop_status in prop_lines:
        st = prop_status.strip().lower()
        result["properties"].append({"name": prop_name, "status": st})
        if st == "proven":
            result["proven"] += 1
        elif st in ("falsified", "cex"):
            result["falsified"] += 1
        else:
            result["undetermined"] += 1

    if not prop_lines:
        result["proven"] = len(re.findall(r"STATUS:\s*proven", raw, re.IGNORECASE))
        result["falsified"] = len(re.findall(r"STATUS:\s*(?:falsified|cex)", raw, re.IGNORECASE))
        result["undetermined"] = len(re.findall(r"STATUS:\s*undetermined", raw, re.IGNORECASE))

    total = result["proven"] + result["falsified"] + result["undetermined"]
    if total > 0 and result["falsified"] == 0:
        result["status"] = "proven" if result["undetermined"] == 0 else "partially_proven"
    elif result["falsified"] > 0:
        result["status"] = "falsified"
    else:
        result["status"] = "no_properties"

    return result


# ---------------------------------------------------------------------------
# SVA injection
# ---------------------------------------------------------------------------

def inject_sva(rtl_code, sva_code, module_name):
    """Inject SVA before the endmodule of the top module."""
    injection = "\n// --- Injected SVA assertions ---\n" + sva_code + "\n\n"

    # Try to find the specific module's endmodule
    pattern = re.compile(
        r"(module\s+" + re.escape(module_name) + r"\b.*?)(endmodule)",
        re.DOTALL,
    )
    m = pattern.search(rtl_code)
    if m:
        return rtl_code[:m.end(1)] + injection + rtl_code[m.start(2):]

    # Fallback: inject before last endmodule
    last = rtl_code.rfind("endmodule")
    if last >= 0:
        return rtl_code[:last] + injection + rtl_code[last:]

    return rtl_code + injection + "\nendmodule\n"


def build_reset_tcl(clock, reset):
    """Build JG clock/reset TCL."""
    tcl = ""
    if clock:
        tcl += f"catch {{clock {clock}}}\n"
    else:
        tcl += "catch {clock clk}\ncatch {clock CLK}\n"

    if reset:
        active_low = any(p in reset.lower() for p in ["_n", "rstn", "resetn", "prestn"])
        if active_low:
            tcl += f"catch {{reset -expression {{!{reset}}}}}\n"
        else:
            tcl += f"catch {{reset {reset}}}\n"
    else:
        tcl += "catch {reset -expression {!reset_}}\ncatch {reset rst}\n"

    return tcl


# ---------------------------------------------------------------------------
# Evaluate one design
# ---------------------------------------------------------------------------

def evaluate_design(design_dir, rtl_code, module_name, clock, reset):
    """Evaluate sva_assertion.sv in design_dir against the design RTL."""
    sva_path = os.path.join(design_dir, "sva_assertion.sv")
    if not os.path.exists(sva_path):
        return None

    with open(sva_path) as f:
        sva_code = f.read().strip()

    if not sva_code:
        return {
            "status": "empty",
            "total": 0, "proven": 0, "falsified": 0, "undetermined": 0,
            "pass_rate": 0.0,
        }

    # Count assertions
    n_assertions = len(re.findall(r"assert\s+property\s*\(", sva_code))

    # Inject and run JG
    combined = inject_sva(rtl_code, sva_code, module_name)
    reset_tcl = build_reset_tcl(clock, reset)

    tcl = (
        f"analyze -sv dut.sv\n"
        f"elaborate -top {module_name}\n"
        f"{reset_tcl}"
        f"prove -all -time_limit 1m\n"
        f'puts "===RESULTS==="\n'
        f'foreach p [get_property_list] {{\n'
        f'    set st [get_status $p]\n'
        f'    puts "PROP: $p STATUS: $st"\n'
        f'}}\n'
        f'puts "===END==="\n'
        f'exit\n'
    )

    try:
        raw = _run_jg_ssh(module_name, combined, tcl, tag="eval_ab", timeout=300)
        parsed = _parse_prove_results(raw)
    except subprocess.TimeoutExpired:
        parsed = {"status": "timeout", "proven": 0, "falsified": 0,
                  "undetermined": 0, "errors": ["timeout"], "raw": ""}
    except Exception as e:
        parsed = {"status": "error", "proven": 0, "falsified": 0,
                  "undetermined": 0, "errors": [str(e)], "raw": ""}

    total = parsed["proven"] + parsed["falsified"] + parsed["undetermined"]
    pass_rate = parsed["proven"] / total if total > 0 else 0.0

    # If JG found syntax/compilation error, count all as errors
    if parsed["status"] in ("syntax_error", "compilation_error"):
        return {
            "status": parsed["status"],
            "total": n_assertions,
            "proven": 0,
            "falsified": 0,
            "undetermined": 0,
            "error_count": n_assertions,
            "pass_rate": 0.0,
            "errors": parsed.get("errors", []),
        }

    return {
        "status": parsed["status"],
        "total": total if total > 0 else n_assertions,
        "proven": parsed["proven"],
        "falsified": parsed["falsified"],
        "undetermined": parsed["undetermined"],
        "error_count": 0,
        "pass_rate": pass_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(args):
    log(f"evaluate_ab: evaluating {args.debug_dir}")

    # Load design info for clock/reset/RTL
    log("Loading test designs for RTL + clock/reset info...")
    all_designs = load_test_designs()
    design_map = {d["design_id"].replace("/", "__"): d for d in all_designs}
    # Also index by raw design_id
    for d in all_designs:
        design_map[d["design_id"]] = d

    # Find all design dirs with sva_assertion.sv
    eval_results = []
    design_dirs = sorted(os.listdir(args.debug_dir))

    for dirname in design_dirs:
        design_path = os.path.join(args.debug_dir, dirname)
        if not os.path.isdir(design_path):
            continue
        sva_path = os.path.join(design_path, "sva_assertion.sv")
        if not os.path.exists(sva_path):
            continue

        # Check for existing eval
        eval_path = os.path.join(design_path, "eval_result.json")
        if args.resume and os.path.exists(eval_path):
            log(f"  [resume] {dirname} already evaluated")
            with open(eval_path) as f:
                eval_results.append({"design_id": dirname, **json.load(f)})
            continue

        # Find design info
        design = design_map.get(dirname)
        if not design:
            log(f"  [skip] {dirname}: no design info found")
            continue

        log(f"  Evaluating {dirname} (module={design['module_name']})...")
        t0 = time.time()
        result = evaluate_design(
            design_path, design["rtl"], design["module_name"],
            design.get("clock"), design.get("reset"),
        )
        elapsed = time.time() - t0

        if result is None:
            log(f"    [skip] no sva_assertion.sv")
            continue

        log(f"    status={result['status']} proven={result['proven']} "
            f"total={result['total']} pass_rate={result['pass_rate']:.3f} "
            f"({elapsed:.1f}s)")

        result["wall_time_s"] = round(elapsed, 1)
        eval_results.append({"design_id": dirname, **result})

        # Save per-design result
        with open(eval_path, "w") as f:
            json.dump(result, f, indent=2)

    # Write CSV
    if eval_results:
        csv_path = args.output or os.path.join(args.debug_dir, "evaluation_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "design_id", "status", "total", "proven", "falsified",
                "undetermined", "error_count", "pass_rate", "wall_time_s",
            ])
            writer.writeheader()
            for r in eval_results:
                row = {k: r.get(k, "") for k in writer.fieldnames}
                writer.writerow(row)
        log(f"\nResults CSV: {csv_path}")

    # Summary
    n = len(eval_results)
    if n > 0:
        avg_pass = sum(r.get("pass_rate", 0) for r in eval_results) / n
        total_proven = sum(r.get("proven", 0) for r in eval_results)
        total_assertions = sum(r.get("total", 0) for r in eval_results)
        n_syntax_ok = sum(1 for r in eval_results
                          if r.get("status") not in ("syntax_error", "compilation_error", "empty"))
        avg_syntax = n_syntax_ok / n

        summary = {
            "total_designs": n,
            "avg_pass_rate": round(avg_pass, 4),
            "avg_syntax": round(avg_syntax, 4),
            "total_proven": total_proven,
            "total_assertions": total_assertions,
            "overall_pass_rate": round(total_proven / max(total_assertions, 1), 4),
        }

        summary_path = os.path.join(args.debug_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        log(f"\n{'='*50}")
        log(f"EVALUATION SUMMARY")
        log(f"  Designs evaluated : {n}")
        log(f"  avg_syntax        : {avg_syntax:.3f}")
        log(f"  avg_pass_rate     : {avg_pass:.3f}")
        log(f"  total_proven      : {total_proven}")
        log(f"  total_assertions  : {total_assertions}")
        log(f"  overall_pass_rate : {summary['overall_pass_rate']:.3f}")
    else:
        log("No designs evaluated.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AssertionBench generated assertions with JasperGold",
    )
    parser.add_argument("--debug_dir", required=True,
                        help="Directory containing generated SVA (baseline or pipeline output)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <debug_dir>/evaluation_results.csv)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip designs with existing eval_result.json")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
