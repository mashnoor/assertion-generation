"""Rebuild FVEval-format CSVs from trial directories and compute pass@k metrics."""

import csv
import json
import os
import sys
from pathlib import Path
from collections import Counter

def comb(n, k):
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    k = min(k, n - k)
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def rebuild_csv(output_dir, dataset_csv, model_name="qwen3.5:35b"):
    """Rebuild FVEval LMResult CSV from trial directories."""
    import pandas as pd

    # Load original dataset for ref_solution and design_rtl
    df = pd.read_csv(dataset_csv)
    design_lookup = {row["task_id"]: row for _, row in df.iterrows()}

    base = Path(output_dir)
    trial_dirs = sorted([d for d in base.iterdir() if d.is_dir() and "_trial_" in d.name])

    rows = []
    for td in trial_dirs:
        parts = td.name.rsplit("_trial_", 1)
        design_name = parts[0]
        trial_id = int(parts[1])

        # Find matching design in dataset
        task_id = None
        for tid in design_lookup:
            if tid == design_name or tid.endswith("/" + design_name):
                task_id = tid
                break

        sva_file = td / "sva_assertion.sv"
        tb_file = td / "packaged_tb.sva"
        rtl_dir = td / (design_name.split("/")[-1] if "/" in design_name else "")

        sva_code = sva_file.read_text() if sva_file.exists() else ""
        packaged_tb = tb_file.read_text() if tb_file.exists() else ""

        # Get design RTL from dataset or from trial dir
        design_rtl = ""
        if task_id and task_id in design_lookup:
            design_rtl = design_lookup[task_id].get("design_rtl", "")
        # Fallback: look for RTL files in subdirectories
        if not design_rtl:
            for sv_file in td.rglob("rtl.sv"):
                design_rtl = sv_file.read_text()
                break

        ref_solution = ""
        if task_id and task_id in design_lookup:
            ref_solution = design_lookup[task_id].get("ref_solution", "")

        exp_id = os.path.basename(dataset_csv).replace(".csv", "")

        rows.append({
            "experiment_id": exp_id,
            "task_id": f"{design_name}_trial_{trial_id}",
            "model_name": model_name,
            "response": sva_code,
            "ref_solution": ref_solution,
            "user_prompt": "",
            "output_tb": packaged_tb,
            "design_rtl": design_rtl,
            "cot_response": "cot_response\n",
        })

    results_df = pd.DataFrame(rows)
    model_tag = model_name.replace(":", "_").replace("/", "_")
    csv_path = os.path.join(output_dir, f"{model_tag}_{exp_id}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Rebuilt CSV: {csv_path} ({len(rows)} rows)")
    return csv_path, len(rows)


def compute_metrics(output_dir):
    """Compute pass@k metrics from agent_result.json files."""
    base = Path(output_dir)
    trial_dirs = sorted([d for d in base.iterdir() if d.is_dir() and "_trial_" in d.name])

    design_results = {}  # design -> list of bools (has_proven_and_no_falsified)
    statuses = Counter()
    syntax_ok_count = 0
    total_count = 0

    for td in trial_dirs:
        parts = td.name.rsplit("_trial_", 1)
        design = parts[0]
        total_count += 1

        rf = td / "agent_result.json"
        if rf.exists():
            r = json.load(open(rf))
            status = r.get("status", "unknown")
            statuses[status] += 1

            proven = r.get("proven", 0)
            falsified = r.get("falsified", 0)
            undetermined = r.get("undetermined", 0)
            total_props = proven + falsified + undetermined

            syntax_ok = status not in ("syntax_error", "compilation_error", "no_result", "")
            if syntax_ok:
                syntax_ok_count += 1

            # "pass" = at least 1 property proven, none falsified
            is_pass = proven > 0 and falsified == 0
            design_results.setdefault(design, []).append(is_pass)
        else:
            statuses["no_result"] += 1
            design_results.setdefault(design, []).append(False)

    n_designs = len(design_results)

    # pass@k
    metrics = {}
    for k in [1, 5]:
        total = 0
        count = 0
        for trials in design_results.values():
            n = len(trials)
            if n >= k:
                c = sum(trials)
                total += 1 - comb(n - c, k) / comb(n, k)
                count += 1
        metrics[f"pass@{k}"] = total / count if count else 0

    # Syntax pass@1
    design_syntax = {}
    for td in trial_dirs:
        parts = td.name.rsplit("_trial_", 1)
        design = parts[0]
        rf = td / "agent_result.json"
        if rf.exists():
            r = json.load(open(rf))
            s = r.get("status", "")
            ok = s not in ("syntax_error", "compilation_error", "no_result", "")
            design_syntax.setdefault(design, []).append(ok)
        else:
            design_syntax.setdefault(design, []).append(False)

    syntax_p1 = sum(
        1 - comb(len(t) - sum(t), 1) / comb(len(t), 1)
        for t in design_syntax.values()
    ) / n_designs

    return {
        "n_designs": n_designs,
        "n_trials": total_count,
        "statuses": dict(statuses.most_common()),
        "syntax_rate": syntax_ok_count / total_count if total_count else 0,
        "syntax_pass@1": syntax_p1,
        **metrics,
    }


if __name__ == "__main__":
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    FVEVAL_DATA = os.environ.get(
        "FVEVAL_DATA",
        os.path.join(_SCRIPT_DIR, "FVEval", "data_design2sva", "data"),
    )
    RESULTS_BASE = os.environ.get(
        "FVEVAL_RESULTS",
        os.path.join(_SCRIPT_DIR, "results"),
    )

    configs = [
        ("Pipeline", f"{RESULTS_BASE}/fveval_qwen3_5_35b_design2sva_pipeline",
         f"{FVEVAL_DATA}/design2sva_pipeline.csv"),
        ("FSM", f"{RESULTS_BASE}/fveval_qwen3_5_35b_design2sva_fsm",
         f"{FVEVAL_DATA}/design2sva_fsm.csv"),
    ]

    print("=" * 70)
    print("FVEval Design2SVA Results — qwen3.5:35b + Tool-Augmented Pipeline")
    print("=" * 70)

    all_metrics = {}
    for name, output_dir, dataset_csv in configs:
        print(f"\n--- {name} ---")

        # Rebuild CSV
        csv_path, n_rows = rebuild_csv(output_dir, dataset_csv)

        # Compute metrics
        m = compute_metrics(output_dir)
        all_metrics[name] = m

        print(f"  Designs: {m['n_designs']}/96, Trials: {m['n_trials']}")
        print(f"  Statuses: {m['statuses']}")
        print(f"  Syntax rate: {m['syntax_rate']:.3f}")
        print(f"  syntax pass@1: {m['syntax_pass@1']:.3f}")
        print(f"  pass@1: {m['pass@1']:.3f}")
        print(f"  pass@5: {m['pass@5']:.3f}")

    # Combined metrics
    print(f"\n{'=' * 70}")
    print("Combined (Pipeline + FSM):")
    # Weighted average
    total_designs = sum(m["n_designs"] for m in all_metrics.values())
    for metric in ["syntax_pass@1", "pass@1", "pass@5"]:
        weighted = sum(
            m[metric] * m["n_designs"] for m in all_metrics.values()
        ) / total_designs
        print(f"  {metric}: {weighted:.3f}")

    # Comparison table — FVEval Table III (Design2SVA)
    # Columns in Table III: Syntax@1, Syntax@5, Func@1, Func@5 (per split)
    # Func = "whether the model has generated an assertion that can be proven"
    print(f"\n{'=' * 70}")
    print("Comparison with FVEval Table III (Design2SVA)")
    print(f"{'=' * 70}")

    p = all_metrics.get("Pipeline", {})
    f = all_metrics.get("FSM", {})

    print(f"\n  Pipeline designs (Func = assertion can be proven):")
    print(f"  {'Method':<35} {'Func@1':>8} {'Func@5':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8}")
    # FVEval Table III numbers (from paper, Design2SVA benchmark)
    pipe_baselines = [
        ("gpt-4o (FVEval)", 0.104, 0.427),
        ("gemini-1.5-pro (FVEval)", 0.175, 0.500),
        ("gemini-1.5-flash (FVEval)", 0.125, 0.498),
    ]
    for method, p1, p5 in pipe_baselines:
        print(f"  {method:<35} {p1:>8.3f} {p5:>8.3f}")
    print(f"  {'Ours (qwen3.5:35b + tools)':<35} {p.get('pass@1',0):>8.3f} {p.get('pass@5',0):>8.3f}")

    print(f"\n  FSM designs:")
    print(f"  {'Method':<35} {'Func@1':>8} {'Func@5':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8}")
    fsm_baselines = [
        ("gpt-4o (FVEval)", 0.373, 0.900),
        ("gemini-1.5-pro (FVEval)", 0.427, 0.906),
    ]
    for method, p1, p5 in fsm_baselines:
        print(f"  {method:<35} {p1:>8.3f} {p5:>8.3f}")
    print(f"  {'Ours (qwen3.5:35b + tools)':<35} {f.get('pass@1',0):>8.3f} {f.get('pass@5',0):>8.3f}")

    p1_comb = sum(m["pass@1"] * m["n_designs"] for m in all_metrics.values()) / total_designs
    p5_comb = sum(m["pass@5"] * m["n_designs"] for m in all_metrics.values()) / total_designs
    print(f"\n  Combined (96+96): Func@1={p1_comb:.3f}, Func@5={p5_comb:.3f}")
