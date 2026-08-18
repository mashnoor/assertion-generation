"""
run_fveval_design2sva.py — Run our tool-augmented pipeline on FVEval Design2SVA benchmark.

Matches FVEval's exact task: given RTL + testbench, generate 1 SVA assertion per trial.
NO NL spec — the agent decides what to verify (same as FVEval baselines).

For each design:
  1. AST-index the RTL into per-design ChromaDB
  2. ReAct agent gathers context via tools (Phase A)
  3. Agent generates 1 SVA assertion (Phase B) with JG verification loop
  4. Output packaged into FVEval testbench format for their evaluator

Output CSV matches FVEval's LMResult format so their evaluator can be used directly:
  python run_evaluation.py -i <output_dir> --task design2sva -m <model>

Usage:
    python run_fveval_design2sva.py \
        --dataset_csv ../FVEval/data_design2sva/data/design2sva_pipeline.csv \
        --provider ollama --model qwen3.5:35b \
        --output_dir results/fveval_d2sva_pipeline/ \
        --num_trials 5 --offset 0 --limit 96
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
import sys
import time

import pandas as pd

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(*args, **kwargs):
    print(f"[{_ts()}]", *args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Lazy imports (avoid slow load on login node)
# ---------------------------------------------------------------------------

def _import_deps():
    from pipeline_v4 import _init_embeddings, _init_llm
    from ast_indexer import ASTIndexer
    from tools import DesignTools
    from agent import SVAAgent
    from langchain_chroma import Chroma
    return {
        "_init_embeddings": _init_embeddings,
        "_init_llm": _init_llm,
        "ASTIndexer": ASTIndexer,
        "DesignTools": DesignTools,
        "SVAAgent": SVAAgent,
        "Chroma": Chroma,
    }


# ---------------------------------------------------------------------------
# FVEval testbench packaging (match their format exactly)
# ---------------------------------------------------------------------------

def package_fveval_testbench(testbench: str, sva_code: str) -> str:
    """Package SVA into FVEval testbench format.

    FVEval expects: tb prefix (up to tb_reset assign) + SVA code + endmodule + bind.
    """
    parts = testbench.split("endmodule")
    prefix = parts[0].rstrip()
    postfix = "endmodule" + "endmodule".join(parts[1:]) if len(parts) > 1 else "endmodule"

    # Ensure tb_reset exists
    if "assign tb_reset" not in prefix:
        prefix += "\nassign tb_reset = (reset_ == 1'b0);\n"

    # Strip markdown fences
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*\n(.*?)```", sva_code, re.DOTALL)
    if m:
        sva_code = m.group(1).strip()

    return prefix + "\n\n" + sva_code + "\n\n" + postfix


# ---------------------------------------------------------------------------
# Design file setup (for DesignTools)
# ---------------------------------------------------------------------------

def setup_design_dir(design_dir, rtl_code, top_module):
    """Write RTL + graph files so DesignTools can find them."""
    mod_dir = os.path.join(design_dir, top_module)
    os.makedirs(mod_dir, exist_ok=True)
    with open(os.path.join(mod_dir, "rtl.sv"), "w") as f:
        f.write(rtl_code)

    module_names = re.findall(r"\bmodule\s+(\w+)", rtl_code)
    # Exclude testbench module names
    module_names = [m for m in module_names if "_tb" not in m]
    graph = {"sorted_modules": module_names or [top_module], "adjacency_list": {}}
    with open(os.path.join(design_dir, "design_graph.json"), "w") as f:
        json.dump(graph, f, indent=2)


def detect_top_module(rtl_code, testbench):
    """Extract the top module name from the bind statement in the testbench."""
    bind_match = re.search(r"bind\s+(\w+)\s+", testbench)
    if bind_match:
        return bind_match.group(1)
    # Fallback: first module in RTL that isn't a testbench
    modules = re.findall(r"\bmodule\s+(\w+)", rtl_code)
    for m in modules:
        if "_tb" not in m:
            return m
    return modules[0] if modules else "top"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class DesignTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise DesignTimeoutError("Timeout")


# ---------------------------------------------------------------------------
# FVEval-style spec: no NL spec, just "generate 1 assertion for this design"
# ---------------------------------------------------------------------------

FVEVAL_SPEC_TEMPLATE = """\
Generate a single, meaningful SVA assertion for the {module} module.
Choose the most important property to verify for this design.
The assertion must use @(posedge clk) disable iff (tb_reset) format.
Output ONLY the assertion code (property + assert statement).
Include any needed modeling code (wires, assigns) before the assertion.
Do NOT instantiate the design module. Do NOT use 'initial' blocks."""


# ---------------------------------------------------------------------------
# Process one design × one trial
# ---------------------------------------------------------------------------

def process_trial(task_id, rtl_code, testbench, trial_id,
                  llm, embeddings, deps, db_base, output_dir,
                  max_verify_rounds=3, timeout_s=600):
    """Run our pipeline on one FVEval design trial. Returns result dict."""
    top_module = detect_top_module(rtl_code, testbench)
    design_dir = os.path.join(output_dir, f"{task_id}_trial_{trial_id}")
    os.makedirs(design_dir, exist_ok=True)

    # Per-trial ChromaDB
    db_path = os.path.join(db_base, f"chroma_{task_id}_{trial_id}")

    # Init ChromaDB + indexer
    vector_store = deps["Chroma"](
        collection_name="rtl_ast_chunks",
        embedding_function=embeddings,
        persist_directory=db_path,
    )
    indexer = deps["ASTIndexer"](
        db_path=db_path,
        embeddings=embeddings,
        collection_name="rtl_ast_chunks",
    )

    # Index design
    try:
        n_chunks = indexer.index_design(task_id, rtl_code, "fveval")
        log(f"    indexed {n_chunks} chunks")
    except Exception as e:
        log(f"    indexing failed: {e}")

    # Setup design files for DesignTools
    setup_design_dir(design_dir, rtl_code, top_module)

    # Init tools
    tools = deps["DesignTools"](
        vector_store=vector_store,
        design_id=task_id,
        design_dir=design_dir,
        llm=llm,
    )

    # Build the FVEval-style spec (no NL spec — agent decides what to verify)
    spec_text = FVEVAL_SPEC_TEMPLATE.format(module=top_module)

    # Run our ReAct agent
    agent = deps["SVAAgent"](
        llm=llm,
        tools=tools,
        spec_id=f"{task_id}_trial_{trial_id}",
        debug_dir=design_dir,
        max_context_rounds=6,
        max_verify_rounds=max_verify_rounds,
    )

    result = agent.run(
        spec_text=spec_text,
        design_id=task_id,
        top_module=top_module,
        timeout_s=timeout_s,
    )

    sva_code = result.sva_code or ""

    # Package into FVEval testbench format
    packaged_tb = package_fveval_testbench(testbench, sva_code)

    # Save
    with open(os.path.join(design_dir, "sva_assertion.sv"), "w") as f:
        f.write(sva_code)
    with open(os.path.join(design_dir, "packaged_tb.sva"), "w") as f:
        f.write(packaged_tb)
    with open(os.path.join(design_dir, "agent_result.json"), "w") as f:
        json.dump({
            "status": result.status,
            "proven": result.proven,
            "falsified": result.falsified,
            "undetermined": result.undetermined,
            "tool_calls": len(result.tool_calls),
            "jg_iterations": result.jg_iterations,
            "wall_time_s": round(result.wall_time_s, 1),
        }, f, indent=2)

    return {
        "sva_code": sva_code,
        "packaged_tb": packaged_tb,
        "design_rtl": rtl_code,
        "status": result.status,
        "wall_time_s": round(result.wall_time_s, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run our pipeline on FVEval Design2SVA benchmark (RTL-only, no NL spec)"
    )
    parser.add_argument("--dataset_csv", required=True,
                        help="Path to design2sva_pipeline.csv or design2sva_fsm.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for FVEval-format results")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openrouter"])
    parser.add_argument("--model", default="qwen3.5:35b")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="Number of trials per design (for pass@k)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db_path", default=None,
                        help="Base path for ChromaDB (default: output_dir/chroma)")
    parser.add_argument("--max_verify", type=int, default=3)
    parser.add_argument("--design_timeout", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    db_base = args.db_path or os.path.join(args.output_dir, "chroma")
    os.makedirs(db_base, exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.dataset_csv)
    log(f"Loaded {len(df)} designs from {args.dataset_csv}")

    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit:
        df = df.iloc[:args.limit]
    log(f"Processing {len(df)} designs (offset={args.offset}, limit={args.limit})")
    log(f"Trials per design: {args.num_trials}")

    # Import deps
    log("Importing pipeline dependencies...")
    deps = _import_deps()

    # Init LLM + embeddings
    log("Initializing LLM + embeddings...")
    embeddings = deps["_init_embeddings"](args.provider)
    llm = deps["_init_llm"](args.provider, args.model)

    # Process each design
    all_results = []
    exp_id = os.path.basename(args.dataset_csv).replace(".csv", "")
    start_time = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        design_name = row["design_name"]
        task_id = row["task_id"]
        rtl = row["prompt"]
        testbench = row["testbench"]

        log(f"\n{'='*60}")
        log(f"Design {idx+1}/{len(df)}: {design_name}/{task_id}")

        for trial in range(args.num_trials):
            trial_key = f"{design_name}_{task_id}_trial_{trial}"

            # Resume check
            trial_dir = os.path.join(args.output_dir, f"{task_id}_trial_{trial}")
            tb_path = os.path.join(trial_dir, "packaged_tb.sva")
            if args.resume and os.path.exists(tb_path):
                log(f"  [resume] trial {trial} already done")
                with open(tb_path) as f:
                    packaged_tb = f.read()
                sva_path = os.path.join(trial_dir, "sva_assertion.sv")
                sva_code = ""
                if os.path.exists(sva_path):
                    with open(sva_path) as f:
                        sva_code = f.read()
                all_results.append({
                    "experiment_id": exp_id,
                    "task_id": trial_key,
                    "model_name": args.model,
                    "response": sva_code,
                    "ref_solution": row.get("ref_solution", ""),
                    "user_prompt": "",
                    "output_tb": packaged_tb,
                    "design_rtl": rtl,
                    "cot_response": "cot_response\n",
                })
                continue

            log(f"  trial {trial}/{args.num_trials}")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(args.design_timeout)
            try:
                result = process_trial(
                    task_id, rtl, testbench, trial,
                    llm, embeddings, deps, db_base, args.output_dir,
                    max_verify_rounds=args.max_verify,
                    timeout_s=args.design_timeout,
                )
            except DesignTimeoutError:
                log(f"  [timeout] trial {trial}")
                result = {"sva_code": "", "packaged_tb": testbench,
                          "design_rtl": rtl, "status": "timeout"}
            except Exception as e:
                log(f"  [error] trial {trial}: {e}")
                result = {"sva_code": "", "packaged_tb": testbench,
                          "design_rtl": rtl, "status": "error"}
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            all_results.append({
                "experiment_id": exp_id,
                "task_id": trial_key,
                "model_name": args.model,
                "response": result.get("sva_code", ""),
                "ref_solution": row.get("ref_solution", ""),
                "user_prompt": "",
                "output_tb": result.get("packaged_tb", ""),
                "design_rtl": result.get("design_rtl", rtl),
                "cot_response": "cot_response\n",
            })

    # Save in FVEval LMResult CSV format
    results_df = pd.DataFrame(all_results)
    model_tag = args.model.replace(":", "_").replace("/", "_")
    csv_path = os.path.join(args.output_dir, f"{model_tag}_{exp_id}.csv")
    results_df.to_csv(csv_path, index=False)

    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"Done in {elapsed/60:.1f} min")
    log(f"Results: {csv_path}")
    log(f"Total: {len(all_results)} ({len(df)} designs x {args.num_trials} trials)")
    log(f"\nTo evaluate with FVEval:")
    log(f"  cd ../FVEval")
    log(f"  python run_evaluation.py -i {os.path.abspath(args.output_dir)} "
        f"--task design2sva -m {model_tag}")


if __name__ == "__main__":
    main()
