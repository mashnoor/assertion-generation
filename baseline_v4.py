# -*- coding: utf-8 -*-
"""
HPC-compatible baseline SVA generation.

Sends raw RTL + NL spec directly to the LLM with no pipeline assistance.
Uses requests-based Ollama API (no langchain) to avoid import hangs on HPC.
Output layout matches pipeline_v4 for apples-to-apples evaluation.
"""

try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import argparse
import os
import json
import re
import signal
import datetime
import time as _time
import requests
import pandas as pd


# ---------------------------------------------------------------------------
# Minimal Ollama LLM (same approach as pipeline_v4.py MinimalLLM)
# ---------------------------------------------------------------------------

class MinimalLLM:
    """Call Ollama /api/chat via requests — no langchain needed."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def invoke(self, system_msg: str, user_msg: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "You are an expert in SystemVerilog formal verification. "
    "Your task is to write a SystemVerilog Assertion (SVA) for the given RTL design and specification. "
    "Output ONLY the SVA code — property declarations and assert statements. "
    "No explanations, no markdown, no comments beyond what is needed."
)

USER_TEMPLATE = (
    "## RTL Design\n```verilog\n{rtl}\n```\n\n"
    "## Assertion Specification\n{spec}\n\n"
    "Write the SVA property and assert statement for this specification."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SpecTimeoutError(Exception):
    pass

def _spec_timeout_handler(signum, frame):
    raise SpecTimeoutError("Spec processing timed out")

def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(*args, **kwargs):
    print(f"[{_ts()}]", *args, **kwargs, flush=True)

def _extract_sva(text: str) -> str:
    m = re.search(r"```(?:systemverilog|sv|verilog)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

def _spec_done(debug_dir: str, design_id: str, spec_id: str) -> bool:
    return os.path.exists(
        os.path.join(debug_dir, design_id, "specs", spec_id, "sva_assertion.sv")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HPC-compatible baseline: raw RTL + spec -> LLM (no langchain)"
    )
    parser.add_argument("--designs_csv", required=True)
    parser.add_argument("--specs_csv", required=True)
    parser.add_argument("--debug_dir", required=True)
    parser.add_argument("--model", default="qwen3.5:35b")
    parser.add_argument("--spec_limit", type=int, default=None)
    parser.add_argument("--spec_timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N designs (for SLURM array parallelism).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max designs to process after offset.")
    parser.add_argument("--design_ids", nargs="+", default=None,
                        help="Only process these design IDs.")
    args = parser.parse_args()

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = MinimalLLM(base_url=base_url, model=args.model)
    log(f"[Baseline] Model: {args.model}  Ollama: {base_url}")
    log(f"[Baseline] Debug dir: {args.debug_dir}")

    designs_df = pd.read_csv(args.designs_csv)
    specs_df = pd.read_csv(args.specs_csv)

    # Filter to specific design IDs if given
    if args.design_ids:
        id_set = set(args.design_ids)
        designs_df = designs_df[designs_df["id"].isin(id_set)]
        log(f"[Baseline] Filtered to {len(designs_df)} designs: {args.design_ids}")
    else:
        if args.offset:
            designs_df = designs_df.iloc[args.offset:]
        if args.limit:
            designs_df = designs_df.head(args.limit)

    specs_by_design = specs_df.groupby("parent_design_id")
    os.makedirs(args.debug_dir, exist_ok=True)

    total_specs = 0
    total_done = 0

    for _, row in designs_df.iterrows():
        design_id = row["id"]
        rtl_code = str(row["rtl"])
        design_type = row["type"]

        log(f"\nDesign: {design_id} ({design_type})")
        design_dir = os.path.join(args.debug_dir, design_id)
        os.makedirs(design_dir, exist_ok=True)

        with open(os.path.join(design_dir, "rtl.sv"), "w") as f:
            f.write(rtl_code)

        if design_id not in specs_by_design.groups:
            log(f"  No specs for {design_id}, skipping.")
            continue

        design_specs = specs_by_design.get_group(design_id)
        if args.spec_limit:
            design_specs = design_specs.head(args.spec_limit)

        log(f"  {len(design_specs)} specs")

        for _, spec_row in design_specs.iterrows():
            spec_id = spec_row["id"]
            spec_text = spec_row["spec"]
            total_specs += 1

            spec_dir = os.path.join(design_dir, "specs", spec_id)
            os.makedirs(spec_dir, exist_ok=True)

            if args.resume and _spec_done(args.debug_dir, design_id, spec_id):
                log(f"    [RESUME] {spec_id}")
                total_done += 1
                continue

            log(f"    Generating: {spec_id}")
            with open(os.path.join(spec_dir, "spec.txt"), "w") as f:
                f.write(spec_text)

            user_msg = USER_TEMPLATE.format(rtl=rtl_code, spec=spec_text)
            with open(os.path.join(spec_dir, "prompt.txt"), "w") as f:
                f.write(user_msg)

            if args.spec_timeout:
                signal.signal(signal.SIGALRM, _spec_timeout_handler)
                signal.alarm(args.spec_timeout)

            try:
                t0 = _time.monotonic()
                raw_output = llm.invoke(SYSTEM_MSG, user_msg)
                elapsed = _time.monotonic() - t0
                log(f"      LLM: {elapsed:.1f}s")

                with open(os.path.join(spec_dir, "llm_response_raw.txt"), "w") as f:
                    f.write(raw_output)

                sva = _extract_sva(raw_output)
                with open(os.path.join(spec_dir, "sva_assertion.sv"), "w") as f:
                    f.write(sva)

                total_done += 1

            except SpecTimeoutError:
                log(f"    [TIMEOUT] {spec_id}")
                with open(os.path.join(spec_dir, "sva_assertion.sv"), "w") as f:
                    f.write("// TIMEOUT\n")
            except Exception as e:
                log(f"    [ERROR] {spec_id}: {e}")
                with open(os.path.join(spec_dir, "sva_assertion.sv"), "w") as f:
                    f.write(f"// ERROR: {e}\n")
            finally:
                signal.alarm(0)

    log(f"\n[Baseline] Done: {total_done}/{total_specs} specs")


if __name__ == "__main__":
    main()
