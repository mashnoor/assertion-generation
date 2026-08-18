"""
pipeline_ab.py — AssertionBench pipeline: tool-augmented assertion generation.

Our approach: for each design, use the cursor_style pipeline to:
  1. Index design AST into ChromaDB (semantic search)
  2. Gather context via tools (module info, always blocks, parameters, flop info)
  3. Generate SVA assertions with rich context + ICL examples
  4. Iteratively verify and fix assertions with JasperGold

This should beat the baseline (direct ICL prompting) because:
  - Tools provide exact signal names, widths, parameters
  - JG verification catches syntax/semantic errors
  - Iterative refinement fixes broken assertions
"""

try:
    import pysqlite3, sys as _sys_sq; _sys_sq.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

import requests

# Path setup — import from parent cursor_style/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CURSOR_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_DIR = os.path.dirname(_CURSOR_DIR)
if _CURSOR_DIR not in sys.path:
    sys.path.insert(0, _CURSOR_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from loader import load_test_designs, load_icl_examples


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(*args, **kwargs):
    print(f"[{_ts()}]", *args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Imports from cursor_style (lazy, after path setup)
# ---------------------------------------------------------------------------

def _import_pipeline_deps():
    """Import cursor_style dependencies. Returns (MinimalLLM, embeddings_factory,
    ASTIndexer, DesignTools, Chroma)."""
    from pipeline_v4 import MinimalLLM, _init_embeddings, _init_llm, OllamaDirectEmbeddings
    from ast_indexer import ASTIndexer
    from tools import DesignTools, _run_jg_ssh, _detect_clock_reset, _parse_prove_results, \
        _extract_sva_code, _inject_sva
    from langchain_chroma import Chroma
    return {
        "MinimalLLM": MinimalLLM,
        "_init_embeddings": _init_embeddings,
        "_init_llm": _init_llm,
        "ASTIndexer": ASTIndexer,
        "DesignTools": DesignTools,
        "Chroma": Chroma,
        "_run_jg_ssh": _run_jg_ssh,
        "_detect_clock_reset": _detect_clock_reset,
        "_parse_prove_results": _parse_prove_results,
        "_extract_sva_code": _extract_sva_code,
        "_inject_sva": _inject_sva,
    }


# ---------------------------------------------------------------------------
# SVA extraction from LLM output
# ---------------------------------------------------------------------------

def extract_sva_code(text: str) -> str:
    """Strip markdown fences and return bare SVA code."""
    pattern = re.compile(r"```(?:systemverilog|sv|verilog)?\s*\n(.*?)```", re.DOTALL)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def count_assertions(sva_text: str) -> int:
    return len(re.findall(r"assert\s+property\s*\(", sva_text))


# ---------------------------------------------------------------------------
# Build clock/reset TCL for arbitrary signal names
# ---------------------------------------------------------------------------

def build_reset_tcl(clock: Optional[str], reset: Optional[str],
                    rtl_code: str = "") -> str:
    """Build JasperGold clock/reset TCL from explicit signal names."""
    tcl = ""
    if clock:
        tcl += f"catch {{clock {clock}}}\n"
    else:
        # Try common names
        tcl += "catch {clock clk}\n"
        tcl += "catch {clock CLK}\n"
        tcl += "catch {clock clock}\n"

    if reset:
        # Guess active-low from naming convention
        active_low = any(p in reset.lower() for p in ["_n", "rst_n", "resetn", "rstn", "prestn"])
        if active_low:
            tcl += f"catch {{reset -expression {{!{reset}}}}}\n"
        else:
            tcl += f"catch {{reset {reset}}}\n"
    else:
        tcl += "catch {reset -expression {!reset_}}\n"
        tcl += "catch {reset rst}\n"
        tcl += "catch {reset reset}\n"

    return tcl


# ---------------------------------------------------------------------------
# JG verification (standalone, doesn't need DesignTools)
# ---------------------------------------------------------------------------

def verify_with_jg(rtl_code: str, sva_code: str, module_name: str,
                   clock: Optional[str], reset: Optional[str],
                   deps, timeout: int = 180) -> Dict:
    """Inject SVA into RTL and run JG prove. Returns parsed result dict."""
    _run_jg_ssh = deps["_run_jg_ssh"]
    _parse_prove_results = deps["_parse_prove_results"]
    _inject_sva = deps["_inject_sva"]

    combined = _inject_sva(rtl_code, sva_code, top_module=module_name)
    reset_tcl = build_reset_tcl(clock, reset, rtl_code)

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
        raw = _run_jg_ssh(module_name, combined, tcl, tag="ab_verify", timeout=timeout)
        return _parse_prove_results(raw)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "proven": 0, "falsified": 0,
                "undetermined": 0, "errors": ["JG timeout"], "raw": ""}
    except Exception as e:
        return {"status": "error", "proven": 0, "falsified": 0,
                "undetermined": 0, "errors": [str(e)], "raw": ""}


# ---------------------------------------------------------------------------
# Pipeline prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert SystemVerilog formal verification engineer.
Your task is to generate formally verifiable SVA assertions for a hardware design.

You will be given:
1. The design's module interface, parameters, and behavioral description
2. Detailed analysis of always blocks, clock/reset behavior, and signal relationships
3. Example assertions from similar designs (for reference format)

Generate 10-20 meaningful, distinct SVA assertions that verify:
- Reset behavior (signals go to known states on reset)
- State machine transitions (if applicable)
- Data path correctness (inputs/outputs relationship)
- Protocol compliance (handshaking, valid/ready, etc.)
- Boundary conditions (counter overflow, FIFO full/empty, etc.)

Output ONLY the SVA code — property declarations and assert statements.
No module wrapper, no explanations. Use the exact signal names from the design.\
"""

CONTEXT_PROMPT_TEMPLATE = """\
## Design: {module_name}

### Module Interface
{module_info}

### Parameters
{parameters}

### Behavioral Analysis (Always Blocks)
{always_blocks}

### Design Hierarchy
{hierarchy}

{icl_section}

Now generate SVA assertions for the {module_name} module.
Use the exact signal names shown in the module interface above.
Clock: {clock}
Reset: {reset}
"""

FIX_PROMPT_TEMPLATE = """\
The following SVA assertions had errors when verified by JasperGold:

## Current SVA (with errors)
```systemverilog
{sva_code}
```

## JasperGold Feedback
Status: {status}
Errors:
{errors}

## Design Context
Module: {module_name}
Clock: {clock}
Reset: {reset}

Fix the assertions. Output ONLY the corrected SVA code.
Common fixes:
- Use correct signal names (check module interface)
- Match bit widths in comparisons
- Use correct clock edge: @(posedge {clock})
- Handle reset correctly: disable iff ({disable_iff})
"""

def format_icl_section(icl_examples):
    """Format ICL examples for the context prompt."""
    if not icl_examples:
        return ""
    parts = ["### Reference Examples (similar designs with proven assertions)"]
    for ex in icl_examples[:3]:  # Use 3 ICL examples in pipeline (context budget)
        parts.append(f"\n**{ex['name']}:**")
        parts.append(f"```verilog\n{ex['rtl']}\n```")
        parts.append(f"Proven assertions:\n```systemverilog\n{ex['assertions_text']}\n```")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class DesignTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise DesignTimeoutError("Design timeout")


# ---------------------------------------------------------------------------
# Pipeline per-design processing
# ---------------------------------------------------------------------------

def process_design(design, llm, vector_store, indexer, icl_examples,
                   debug_dir, deps, max_verify_rounds=3, timeout_s=900):
    """Process a single design through the full pipeline.

    Returns a result dict with status, proven, falsified, etc.
    """
    did = design["design_id"]
    module_name = design["module_name"]
    clock = design.get("clock") or "clk"
    reset = design.get("reset")
    rtl = design["rtl"]

    design_dir = os.path.join(debug_dir, did.replace("/", "__"))
    os.makedirs(design_dir, exist_ok=True)

    DesignTools = deps["DesignTools"]

    # ---- Phase 0: Index design AST ----
    if not indexer.is_design_indexed(did):
        log(f"  [phase0] indexing design...")
        try:
            n_chunks = indexer.index_design(did, rtl, "unknown")
            log(f"  [phase0] indexed {n_chunks} chunks")
        except Exception as e:
            log(f"  [phase0] indexing failed: {e}")
    else:
        log(f"  [phase0] already indexed")

    # ---- Save module RTLs for DesignTools ----
    _save_design_files(design_dir, rtl, module_name)

    # ---- Phase A: Gather context via tools ----
    log(f"  [phaseA] gathering context via tools...")
    tools = DesignTools(
        vector_store=vector_store,
        design_id=did,
        design_dir=design_dir,
        llm=llm,
    )

    # Deterministic tool calls (no ReAct — just call what we need)
    context = {}

    try:
        context["hierarchy"] = tools.get_hierarchy()
    except Exception as e:
        context["hierarchy"] = f"(unavailable: {e})"

    try:
        context["module_info"] = tools.get_module_info(module_name)
    except Exception as e:
        context["module_info"] = f"(unavailable: {e})"

    try:
        context["parameters"] = tools.get_parameters(module_name)
    except Exception as e:
        context["parameters"] = f"(unavailable: {e})"

    # Get always blocks for all signals in top module
    try:
        context["always_blocks"] = tools.get_always_blocks("", module_name)
    except Exception as e:
        context["always_blocks"] = f"(unavailable: {e})"

    # Save context
    with open(os.path.join(design_dir, "context.json"), "w") as f:
        json.dump(context, f, indent=2, default=str)

    # ---- Phase B: Generate assertions ----
    log(f"  [phaseB] generating assertions...")

    icl_section = format_icl_section(icl_examples)
    disable_iff = f"!{reset}" if reset and any(p in reset.lower() for p in ["_n", "rstn", "resetn", "prestn"]) else (reset if reset else "!reset_")

    user_prompt = CONTEXT_PROMPT_TEMPLATE.format(
        module_name=module_name,
        module_info=context.get("module_info", "(none)"),
        parameters=context.get("parameters", "(none)"),
        always_blocks=context.get("always_blocks", "(none)"),
        hierarchy=context.get("hierarchy", "(none)"),
        icl_section=icl_section,
        clock=clock,
        reset=reset or "(none detected)",
    )

    # Save prompt
    with open(os.path.join(design_dir, "generation_prompt.txt"), "w") as f:
        f.write(f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_prompt}")

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = llm.invoke(messages)
        sva_code = extract_sva_code(response.content)
    except Exception as e:
        log(f"  [phaseB] LLM call failed: {e}")
        return {
            "design_id": did, "module_name": module_name,
            "status": "error", "error": str(e),
            "total_generated": 0, "proven": 0, "falsified": 0,
            "undetermined": 0, "jg_iterations": 0,
        }

    # Save initial SVA
    with open(os.path.join(design_dir, "sva_v0.sv"), "w") as f:
        f.write(sva_code)

    # ---- Phase C: Verify + iterative fix ----
    log(f"  [phaseC] verifying with JasperGold...")
    jg_iterations = 0

    for iteration in range(max_verify_rounds + 1):
        jg_result = verify_with_jg(rtl, sva_code, module_name, clock, reset, deps)
        jg_iterations += 1

        status = jg_result.get("status", "unknown")
        proven = jg_result.get("proven", 0)
        falsified = jg_result.get("falsified", 0)
        undetermined = jg_result.get("undetermined", 0)
        total = proven + falsified + undetermined

        log(f"    [verify v{iteration}] status={status} proven={proven} "
            f"falsified={falsified} undetermined={undetermined}")

        # Save verification result
        with open(os.path.join(design_dir, f"verification_v{iteration}.json"), "w") as f:
            json.dump(jg_result, f, indent=2, default=str)

        # If proven or no errors to fix, stop
        if status in ("proven", "partially_proven"):
            break
        if iteration >= max_verify_rounds:
            break
        if status in ("timeout", "error") and not jg_result.get("errors"):
            break

        # Try to fix
        log(f"    [fix v{iteration+1}] asking LLM to fix...")
        errors_text = "\n".join(jg_result.get("errors", [])[:5])
        fix_prompt = FIX_PROMPT_TEMPLATE.format(
            sva_code=sva_code,
            status=status,
            errors=errors_text or "(no specific error messages)",
            module_name=module_name,
            clock=clock,
            reset=reset or "rst",
            disable_iff=disable_iff,
        )

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": fix_prompt},
            ]
            fix_response = llm.invoke(messages)
            sva_code = extract_sva_code(fix_response.content)
            with open(os.path.join(design_dir, f"sva_v{iteration+1}.sv"), "w") as f:
                f.write(sva_code)
        except Exception as e:
            log(f"    [fix] LLM fix call failed: {e}")
            break

    # Save final SVA
    with open(os.path.join(design_dir, "sva_assertion.sv"), "w") as f:
        f.write(sva_code)

    n_assertions = count_assertions(sva_code)

    return {
        "design_id": did,
        "module_name": module_name,
        "clock": clock,
        "reset": reset,
        "status": status,
        "total_generated": n_assertions,
        "proven": proven,
        "falsified": falsified,
        "undetermined": undetermined,
        "ground_truth_count": design.get("ground_truth_count", 0),
        "jg_iterations": jg_iterations,
    }


def _save_design_files(design_dir, rtl_code, module_name):
    """Write RTL files to design_dir for DesignTools to find."""
    try:
        import hierarchy as _hier
        hierarchy_obj = _hier.decompose_design(module_name, rtl_code)
        for mod_name, mod_info in hierarchy_obj.modules.items():
            mod_dir = os.path.join(design_dir, mod_name)
            os.makedirs(mod_dir, exist_ok=True)
            with open(os.path.join(mod_dir, "rtl.sv"), "w") as f:
                f.write(mod_info.code)
        graph = {
            "sorted_modules": list(hierarchy_obj.sorted_modules),
            "adjacency_list": {k: list(v) for k, v in hierarchy_obj.adjacency_list.items()},
        }
        with open(os.path.join(design_dir, "design_graph.json"), "w") as f:
            json.dump(graph, f, indent=2)
    except Exception:
        # Fallback: write flat RTL
        mod_dir = os.path.join(design_dir, module_name)
        os.makedirs(mod_dir, exist_ok=True)
        with open(os.path.join(mod_dir, "rtl.sv"), "w") as f:
            f.write(rtl_code)
        module_names = re.findall(r"\bmodule\s+(\w+)", rtl_code)
        graph = {"sorted_modules": module_names or [module_name], "adjacency_list": {}}
        with open(os.path.join(design_dir, "design_graph.json"), "w") as f:
            json.dump(graph, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(args):
    log(f"pipeline_ab starting — provider={args.provider} model={args.model}")
    log(f"  debug_dir={args.debug_dir}")
    log(f"  max_verify_rounds={args.max_verify}")

    # Load dataset
    log("Loading ICL examples...")
    icl_examples = load_icl_examples(k=5)
    log(f"  Loaded {len(icl_examples)} ICL examples")

    log("Loading test designs...")
    designs = load_test_designs()
    log(f"  Loaded {len(designs)} test designs")

    # Apply filters
    if args.design_ids:
        id_set = set(args.design_ids)
        designs = [d for d in designs if d["design_id"] in id_set]
    else:
        if args.offset:
            designs = designs[args.offset:]
        if args.limit:
            designs = designs[:args.limit]

    log(f"  Processing {len(designs)} designs")

    # Import pipeline dependencies
    log("Importing pipeline dependencies...")
    deps = _import_pipeline_deps()

    # Init shared resources
    log("Initialising embeddings + ChromaDB...")
    embeddings = deps["_init_embeddings"](args.provider)
    vector_store = deps["Chroma"](
        collection_name="rtl_ast_chunks",
        embedding_function=embeddings,
        persist_directory=args.db_path,
    )

    log("Initialising LLM...")
    llm = deps["_init_llm"](args.provider, args.model)

    indexer = deps["ASTIndexer"](
        db_path=args.db_path,
        embeddings=embeddings,
        collection_name="rtl_ast_chunks",
    )

    os.makedirs(args.debug_dir, exist_ok=True)

    # Process designs
    results = []
    pipeline_start = time.time()

    for idx, design in enumerate(designs):
        did = design["design_id"]
        log(f"\n{'='*60}")
        log(f"Design {idx+1}/{len(designs)}: {did}  module={design['module_name']}")

        design_dir = os.path.join(args.debug_dir, did.replace("/", "__"))

        # Resume check
        sva_path = os.path.join(design_dir, "sva_assertion.sv")
        if args.resume and os.path.exists(sva_path):
            log(f"  [resume] already done — skipping")
            result_path = os.path.join(design_dir, "result.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    results.append(json.load(f))
            continue

        t0 = time.time()

        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.design_timeout)

        try:
            result = process_design(
                design, llm, vector_store, indexer, icl_examples,
                args.debug_dir, deps,
                max_verify_rounds=args.max_verify,
                timeout_s=args.design_timeout,
            )
        except DesignTimeoutError:
            log(f"  [timeout] design timed out after {args.design_timeout}s")
            result = {
                "design_id": did, "module_name": design["module_name"],
                "status": "timeout", "total_generated": 0,
                "proven": 0, "falsified": 0, "undetermined": 0,
                "jg_iterations": 0,
            }
        except Exception as e:
            log(f"  [error] pipeline failed: {e}")
            result = {
                "design_id": did, "module_name": design["module_name"],
                "status": "error", "error": str(e),
                "total_generated": 0, "proven": 0, "falsified": 0,
                "undetermined": 0, "jg_iterations": 0,
            }
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        result["wall_time_s"] = round(time.time() - t0, 1)
        results.append(result)

        # Save result
        os.makedirs(design_dir, exist_ok=True)
        with open(os.path.join(design_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)

        log(f"  [DONE] status={result['status']} proven={result.get('proven',0)} "
            f"total={result.get('total_generated',0)} time={result['wall_time_s']}s")

    # Summary
    total_time = time.time() - pipeline_start
    total_gen = sum(r.get("total_generated", 0) for r in results)
    total_proven = sum(r.get("proven", 0) for r in results)

    log(f"\n{'='*60}")
    log(f"Pipeline complete in {total_time/60:.1f} min")
    log(f"  Designs: {len(results)}")
    log(f"  Total assertions: {total_gen}")
    log(f"  Total proven: {total_proven}")
    if total_gen > 0:
        log(f"  Overall pass rate: {total_proven/total_gen:.3f}")

    # Save metadata
    metadata = {
        "model": args.model,
        "provider": args.provider,
        "total_designs": len(results),
        "total_assertions": total_gen,
        "total_proven": total_proven,
        "total_time_s": round(total_time, 1),
        "results": results,
    }
    with open(os.path.join(args.debug_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AssertionBench pipeline: tool-augmented assertion generation",
    )
    parser.add_argument("--debug_dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--db_path",
                        default=os.path.join(_THIS_DIR, "chroma_db_ab"),
                        help="ChromaDB persist directory")
    parser.add_argument("--provider", choices=["ollama", "openrouter", "vllm", "hf"],
                        default="ollama")
    parser.add_argument("--model", default="qwen3.5:35b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--design_timeout", type=int, default=900,
                        help="Max seconds per design (default: 900)")
    parser.add_argument("--max_verify", type=int, default=3,
                        help="Max JG verification rounds (default: 3)")
    parser.add_argument("--design_ids", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
