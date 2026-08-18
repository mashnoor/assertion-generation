#!/usr/bin/env python3
"""
export_results.py — Comprehensive results exporter for research paper.

Reads evaluation results, pipeline logs, and agent traces from one or more
result directories and produces:
  1. Per-spec detailed CSV (all metrics, tool calls, timing, etc.)
  2. Per-design aggregated CSV
  3. Aggregate summary JSON
  4. Design complexity breakdown (by type, module count, RTL size)
  5. Comparison tables (pipeline vs baseline, ablation variants)
  6. LaTeX-ready tables

Usage:
    python export_results.py --results_dirs dir1 dir2 ... --labels "Pipeline" "Baseline" \
        --designs_csv ../designs.csv --output_dir paper_results/
"""

try:
    import pysqlite3, sys; sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _count_assertions_in_sv(path: str) -> int:
    """Count assertion property statements in an SVA file."""
    try:
        with open(path) as f:
            text = f.read()
        return len(re.findall(r'\bassert\s+property\b', text, re.IGNORECASE))
    except Exception:
        return 0


def _collect_spec_data(results_dir: str, designs_df: Optional[pd.DataFrame] = None) -> List[dict]:
    """Walk a results directory and collect per-spec data."""
    rows = []
    if not os.path.isdir(results_dir):
        return rows

    for design_id in sorted(os.listdir(results_dir)):
        design_dir = os.path.join(results_dir, design_id)
        specs_dir = os.path.join(design_dir, "specs")
        if not os.path.isdir(specs_dir):
            continue

        # Design-level info
        design_type = ""
        rtl_chars = 0
        module_count = 0
        if designs_df is not None and design_id in designs_df.index:
            design_type = str(designs_df.loc[design_id, "type"])
            rtl_chars = len(str(designs_df.loc[design_id, "rtl"]))

        # Count modules from design_graph.json or module dirs
        graph_path = os.path.join(design_dir, "design_graph.json")
        graph = _load_json(graph_path)
        if graph:
            module_count = len(graph.get("sorted_modules", []))
        else:
            # Count subdirs that have rtl.sv
            module_count = sum(
                1 for d in os.listdir(design_dir)
                if os.path.isfile(os.path.join(design_dir, d, "rtl.sv"))
            ) if os.path.isdir(design_dir) else 0

        for spec_id in sorted(os.listdir(specs_dir)):
            spec_dir = os.path.join(specs_dir, spec_id)
            if not os.path.isdir(spec_dir):
                continue

            row = {
                "design_id": design_id,
                "spec_id": spec_id,
                "design_type": design_type,
                "rtl_chars": rtl_chars,
                "module_count": module_count,
            }

            # Spec text
            spec_path = os.path.join(spec_dir, "spec.txt")
            if os.path.exists(spec_path):
                with open(spec_path) as f:
                    row["spec_text"] = f.read().strip()
            else:
                row["spec_text"] = ""

            # Agent result.json (pipeline runs)
            result = _load_json(os.path.join(spec_dir, "result.json"))
            if result:
                row["status"] = result.get("status", "unknown")
                row["proven"] = result.get("proven", 0)
                row["falsified"] = result.get("falsified", 0)
                row["undetermined"] = result.get("undetermined", 0)
                row["total_props"] = result.get("total", 0)
                row["jg_iterations"] = result.get("jg_iterations", 0)
                row["context_rounds"] = result.get("context_rounds", 0)
                row["wall_time_s"] = result.get("wall_time_s", 0.0)
                row["vacuity_status"] = result.get("vacuity_status", "not_checked")
                row["error_message"] = result.get("error_message", "")

            # Evaluation metrics (from evaluate_v4.py)
            eval_metrics = _load_json(os.path.join(spec_dir, "eval_metrics.json"))
            if eval_metrics:
                row["eval_syntax"] = eval_metrics.get("syntax", 0.0)
                row["eval_functionality"] = eval_metrics.get("functionality", 0.0)
                row["eval_func_relaxed"] = eval_metrics.get("func_relaxed", 0.0)
                row["eval_proven"] = eval_metrics.get("proven", 0)
                row["eval_falsified"] = eval_metrics.get("falsified", 0)
                row["eval_undetermined"] = eval_metrics.get("undetermined", 0)

            # Count assertions in SVA file
            sva_path = os.path.join(spec_dir, "sva_assertion.sv")
            row["num_assertions"] = _count_assertions_in_sv(sva_path)
            row["has_sva"] = os.path.exists(sva_path)

            # Count verification iterations (from verification_v*.json files)
            verify_files = sorted(glob.glob(os.path.join(spec_dir, "verification_v*.json")))
            row["verify_iterations"] = len(verify_files)

            # Tool call log (from agent trace)
            agent_log = _load_json(os.path.join(spec_dir, "agent_trace.json"))
            if agent_log and "tool_calls" in agent_log:
                tools = agent_log["tool_calls"]
                row["total_tool_calls"] = len(tools)
                tool_names = [t.get("tool", "") for t in tools]
                row["tool_call_list"] = ",".join(tool_names)
                # Phase A vs Phase B tool counts
                row["phase_a_tools"] = sum(1 for t in tools if t.get("phase") == "A")
                row["phase_b_tools"] = sum(1 for t in tools if t.get("phase") == "B")
            else:
                row["total_tool_calls"] = row.get("context_rounds", 0) + row.get("jg_iterations", 0)
                row["tool_call_list"] = ""
                row["phase_a_tools"] = 0
                row["phase_b_tools"] = 0

            # Fill defaults for missing fields
            for k in ["status", "proven", "falsified", "undetermined", "total_props",
                       "jg_iterations", "context_rounds", "wall_time_s",
                       "vacuity_status", "error_message",
                       "eval_syntax", "eval_functionality", "eval_func_relaxed",
                       "eval_proven", "eval_falsified", "eval_undetermined"]:
                row.setdefault(k, 0 if k not in ("status", "vacuity_status", "error_message") else "")

            rows.append(row)

    return rows


def _aggregate_by_design(spec_rows: List[dict]) -> List[dict]:
    """Aggregate per-spec rows into per-design summaries."""
    by_design = defaultdict(list)
    for r in spec_rows:
        by_design[r["design_id"]].append(r)

    design_rows = []
    for design_id, specs in sorted(by_design.items()):
        n = len(specs)
        d = {
            "design_id": design_id,
            "design_type": specs[0].get("design_type", ""),
            "rtl_chars": specs[0].get("rtl_chars", 0),
            "module_count": specs[0].get("module_count", 0),
            "num_specs": n,
        }
        # Metrics from eval
        eval_syn = [s["eval_syntax"] for s in specs if isinstance(s.get("eval_syntax"), (int, float))]
        eval_func = [s["eval_functionality"] for s in specs if isinstance(s.get("eval_functionality"), (int, float))]
        d["avg_syntax"] = statistics.mean(eval_syn) if eval_syn else 0.0
        d["avg_functionality"] = statistics.mean(eval_func) if eval_func else 0.0

        # Counts
        d["total_proven"] = sum(s.get("eval_proven", 0) for s in specs)
        d["total_falsified"] = sum(s.get("eval_falsified", 0) for s in specs)
        d["total_undetermined"] = sum(s.get("eval_undetermined", 0) for s in specs)

        # Agent stats
        d["avg_tool_calls"] = statistics.mean([s.get("total_tool_calls", 0) for s in specs])
        d["avg_jg_iterations"] = statistics.mean([s.get("jg_iterations", 0) for s in specs])
        d["avg_wall_time_s"] = statistics.mean([s.get("wall_time_s", 0) for s in specs])
        d["total_wall_time_s"] = sum(s.get("wall_time_s", 0) for s in specs)

        design_rows.append(d)

    return design_rows


def _complexity_bucket(rtl_chars: int, module_count: int) -> str:
    """Categorize designs by complexity."""
    if module_count <= 1 and rtl_chars < 2000:
        return "small"
    elif module_count <= 3 and rtl_chars < 5000:
        return "medium"
    else:
        return "large"


def _generate_latex_table(comparison: List[dict], caption: str = "") -> str:
    """Generate a LaTeX table from comparison data."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")

    # Determine columns
    cols = list(comparison[0].keys()) if comparison else []
    col_spec = "l" + "r" * (len(cols) - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    header = " & ".join(c.replace("_", r"\_") for c in cols) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for row in comparison:
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export detailed results for research paper")
    parser.add_argument("--results_dirs", nargs="+", required=True,
                        help="Result directories to analyze")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each results_dir (e.g., 'Pipeline' 'Baseline')")
    parser.add_argument("--designs_csv", default=None,
                        help="Path to designs.csv for design metadata")
    parser.add_argument("--output_dir", default="paper_results",
                        help="Output directory for exported files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels = args.labels or [f"run_{i}" for i in range(len(args.results_dirs))]
    assert len(labels) == len(args.results_dirs), "labels must match results_dirs"

    # Load designs metadata
    designs_df = None
    if args.designs_csv and os.path.exists(args.designs_csv):
        designs_df = pd.read_csv(args.designs_csv).set_index("id")

    # Collect data for each run
    all_runs = {}
    for label, rdir in zip(labels, args.results_dirs):
        print(f"Collecting data from: {rdir} (label={label})")
        spec_rows = _collect_spec_data(rdir, designs_df)
        all_runs[label] = spec_rows
        print(f"  Found {len(spec_rows)} specs across {len(set(r['design_id'] for r in spec_rows))} designs")

    # -----------------------------------------------------------------------
    # 1. Per-spec detailed CSV for each run
    # -----------------------------------------------------------------------
    for label, spec_rows in all_runs.items():
        out_path = os.path.join(args.output_dir, f"per_spec_{label}.csv")
        if spec_rows:
            df = pd.DataFrame(spec_rows)
            df.to_csv(out_path, index=False)
            print(f"  -> {out_path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # 2. Per-design aggregated CSV for each run
    # -----------------------------------------------------------------------
    for label, spec_rows in all_runs.items():
        design_rows = _aggregate_by_design(spec_rows)
        out_path = os.path.join(args.output_dir, f"per_design_{label}.csv")
        if design_rows:
            df = pd.DataFrame(design_rows)
            df.to_csv(out_path, index=False)
            print(f"  -> {out_path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # 3. Comparison summary table
    # -----------------------------------------------------------------------
    comparison = []
    for label, spec_rows in all_runs.items():
        n = len(spec_rows)
        if n == 0:
            continue

        eval_syn = [r["eval_syntax"] for r in spec_rows if isinstance(r.get("eval_syntax"), (int, float))]
        eval_func = [r["eval_functionality"] for r in spec_rows if isinstance(r.get("eval_functionality"), (int, float))]

        entry = {
            "method": label,
            "num_specs": n,
            "num_designs": len(set(r["design_id"] for r in spec_rows)),
            "avg_syntax": statistics.mean(eval_syn) if eval_syn else 0.0,
            "avg_functionality": statistics.mean(eval_func) if eval_func else 0.0,
            "total_proven": sum(r.get("eval_proven", 0) for r in spec_rows),
            "total_falsified": sum(r.get("eval_falsified", 0) for r in spec_rows),
            "total_undetermined": sum(r.get("eval_undetermined", 0) for r in spec_rows),
            "avg_tool_calls": statistics.mean([r.get("total_tool_calls", 0) for r in spec_rows]),
            "avg_jg_iterations": statistics.mean([r.get("jg_iterations", 0) for r in spec_rows]),
            "avg_wall_time_s": statistics.mean([r.get("wall_time_s", 0) for r in spec_rows]),
        }
        comparison.append(entry)

    if comparison:
        out_path = os.path.join(args.output_dir, "comparison_summary.csv")
        pd.DataFrame(comparison).to_csv(out_path, index=False)
        print(f"\n  -> {out_path}")

        # JSON version
        json_path = os.path.join(args.output_dir, "comparison_summary.json")
        with open(json_path, "w") as f:
            json.dump(comparison, f, indent=2)

    # -----------------------------------------------------------------------
    # 4. Breakdown by design type and complexity
    # -----------------------------------------------------------------------
    for label, spec_rows in all_runs.items():
        if not spec_rows:
            continue

        # By design type (fsm vs pipeline)
        by_type = defaultdict(list)
        for r in spec_rows:
            by_type[r.get("design_type", "unknown")].append(r)

        type_summary = []
        for dtype, rows in sorted(by_type.items()):
            syn = [r["eval_syntax"] for r in rows if isinstance(r.get("eval_syntax"), (int, float))]
            func = [r["eval_functionality"] for r in rows if isinstance(r.get("eval_functionality"), (int, float))]
            type_summary.append({
                "method": label,
                "design_type": dtype,
                "num_specs": len(rows),
                "avg_syntax": statistics.mean(syn) if syn else 0.0,
                "avg_functionality": statistics.mean(func) if func else 0.0,
                "total_proven": sum(r.get("eval_proven", 0) for r in rows),
                "total_falsified": sum(r.get("eval_falsified", 0) for r in rows),
            })

        out_path = os.path.join(args.output_dir, f"by_type_{label}.csv")
        pd.DataFrame(type_summary).to_csv(out_path, index=False)

        # By complexity bucket
        for r in spec_rows:
            r["complexity"] = _complexity_bucket(r.get("rtl_chars", 0), r.get("module_count", 0))

        by_complexity = defaultdict(list)
        for r in spec_rows:
            by_complexity[r["complexity"]].append(r)

        complexity_summary = []
        for bucket in ["small", "medium", "large"]:
            rows = by_complexity.get(bucket, [])
            if not rows:
                continue
            syn = [r["eval_syntax"] for r in rows if isinstance(r.get("eval_syntax"), (int, float))]
            func = [r["eval_functionality"] for r in rows if isinstance(r.get("eval_functionality"), (int, float))]
            complexity_summary.append({
                "method": label,
                "complexity": bucket,
                "num_specs": len(rows),
                "num_designs": len(set(r["design_id"] for r in rows)),
                "avg_syntax": statistics.mean(syn) if syn else 0.0,
                "avg_functionality": statistics.mean(func) if func else 0.0,
            })

        out_path = os.path.join(args.output_dir, f"by_complexity_{label}.csv")
        pd.DataFrame(complexity_summary).to_csv(out_path, index=False)

    # -----------------------------------------------------------------------
    # 5. LaTeX tables
    # -----------------------------------------------------------------------
    if comparison:
        latex_main = _generate_latex_table(
            [{k: v for k, v in c.items() if k in ("method", "num_specs", "avg_syntax", "avg_functionality",
                                                     "total_proven", "total_falsified")}
             for c in comparison],
            caption="Comparison of SVA generation approaches"
        )
        latex_path = os.path.join(args.output_dir, "table_comparison.tex")
        with open(latex_path, "w") as f:
            f.write(latex_main)
        print(f"  -> {latex_path}")

    # -----------------------------------------------------------------------
    # 6. Print summary to stdout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for c in comparison:
        print(f"\n  {c['method']}:")
        print(f"    Specs: {c['num_specs']}  Designs: {c['num_designs']}")
        print(f"    avg_syntax:        {c['avg_syntax']:.4f}")
        print(f"    avg_functionality: {c['avg_functionality']:.4f}")
        print(f"    proven/falsified:  {c['total_proven']}/{c['total_falsified']}")
        print(f"    avg_tool_calls:    {c['avg_tool_calls']:.1f}")
        print(f"    avg_jg_iterations: {c['avg_jg_iterations']:.1f}")
        print(f"    avg_wall_time:     {c['avg_wall_time_s']:.1f}s")

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
