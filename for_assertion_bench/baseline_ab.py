"""
baseline_ab.py — AssertionBench baseline: k-shot ICL assertion generation.

Mimics the AssertionBench paper approach exactly:
  System: "You are an expert in SVA. Generate assertions for the given design."
  User: k ICL examples (design + proven assertions) + test design
  Output: List of SVA assertions

No tools, no RAG, no verification loop.
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

import requests

from loader import load_test_designs, load_icl_examples

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(*args, **kwargs):
    print(f"[{_ts()}]", *args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# MinimalLLM
# ---------------------------------------------------------------------------

class MinimalLLM:
    def __init__(self, base_url, model, temperature=0.7, timeout=600,
                 api_key="", native_ollama=True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.api_key = api_key
        self.native_ollama = native_ollama

    def generate(self, system_msg, user_msg):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if self.native_ollama:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        else:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.temperature,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# HF Transformers LLM (local inference)
# ---------------------------------------------------------------------------

class HFLocalLLM:
    """HuggingFace transformers-based LLM for local inference.

    Qwen3.5 is natively multimodal (Qwen3_5ForConditionalGeneration),
    so we must use AutoProcessor (not AutoTokenizer) and flash-linear-attention
    for fast Gated DeltaNet kernels.
    """
    def __init__(self, model_name, temperature=1.0, max_new_tokens=4096):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        log(f"Loading HF model: {model_name}...")
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        log(f"HF model loaded: {model_name}")

    def generate(self, system_msg, user_msg):
        import torch
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                top_p=0.95 if self.temperature > 0 else None,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _init_hf_llm(model_name):
    return HFLocalLLM(model_name, temperature=1.0)


# ---------------------------------------------------------------------------
# Prompt (mirrors AssertionBench paper)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in SystemVerilog Assertions (SVA).
Your task is to generate a list of formally verifiable assertions for the given \
Verilog hardware design.
Each assertion should be a complete SVA assert property statement.
Generate only the list of assertions with no additional text or explanations.\
"""


def build_user_prompt(icl_examples, test_rtl):
    """Build the user prompt with k-shot ICL examples + test design."""
    parts = []
    for i, ex in enumerate(icl_examples, 1):
        parts.append(f"Program {i}:\n{ex['rtl']}\n")
        parts.append(f"Assertions {i}:\n{ex['assertions_text']}\n")

    parts.append(f"Test Program:\n{test_rtl}\n")
    parts.append("Test Assertions:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SVA extraction from LLM output
# ---------------------------------------------------------------------------

def extract_assertions(llm_output):
    """Parse LLM output and extract SVA assert property statements.

    Returns a list of assertion strings and the full cleaned SVA text.
    """
    # Strip markdown fences if present
    pattern = re.compile(r"```(?:systemverilog|sv|verilog)?\s*\n(.*?)```", re.DOTALL)
    m = pattern.search(llm_output)
    if m:
        text = m.group(1).strip()
    else:
        text = llm_output.strip()

    # Find all assert property(...) statements
    assertions = re.findall(
        r"(?:\w+\s*:\s*)?assert\s+property\s*\(.*?\)\s*;",
        text,
        re.DOTALL,
    )

    # If no structured assertions found, try to extract bare property expressions
    if not assertions:
        # Try lines with |-> or ##
        lines = [l.strip() for l in text.splitlines() if "|->" in l or "##" in l]
        for line in lines:
            line = line.rstrip(";").strip()
            if line:
                assertions.append(f"assert property(@(posedge clk) {line});")

    return assertions, text


def wrap_assertions_sv(assertions, clock="clk"):
    """Wrap extracted assertions into a proper SVA format for JG injection."""
    lines = []
    for i, a in enumerate(assertions):
        # If assertion already has @(posedge ...), keep as-is
        if "@(" in a:
            lines.append(a)
        else:
            # Wrap bare property expression
            expr = a.replace("assert property(", "").rstrip(");").strip()
            lines.append(f"assert property(@(posedge {clock}) {expr});")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class DesignTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise DesignTimeoutError("Design timeout")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(args):
    log(f"baseline_ab starting — provider={args.provider} model={args.model}")
    log(f"  k_shot={args.k_shot}")
    log(f"  debug_dir={args.debug_dir}")

    # Load dataset
    log("Loading ICL examples...")
    icl_examples = load_icl_examples(k=args.k_shot)
    log(f"  Loaded {len(icl_examples)} ICL examples: {[e['name'] for e in icl_examples]}")

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

    # Init LLM
    if args.provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = MinimalLLM(base_url=base_url, model=args.model, temperature=1.0,
                         native_ollama=True)
    elif args.provider == "vllm":
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        llm = MinimalLLM(base_url=base_url, model=args.model, temperature=1.0,
                         api_key="dummy", native_ollama=False)
    elif args.provider == "hf":
        llm = _init_hf_llm(args.model)
    else:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        llm = MinimalLLM(base_url=base_url, model=args.model, temperature=1.0,
                         api_key=api_key, native_ollama=False)

    os.makedirs(args.debug_dir, exist_ok=True)

    # Process each design
    results = []
    for idx, design in enumerate(designs):
        did = design["design_id"]
        log(f"\n{'='*60}")
        log(f"Design {idx+1}/{len(designs)}: {did}  module={design['module_name']}")

        design_dir = os.path.join(args.debug_dir, did.replace("/", "__"))
        os.makedirs(design_dir, exist_ok=True)

        # Resume check
        sva_path = os.path.join(design_dir, "sva_assertion.sv")
        if args.resume and os.path.exists(sva_path):
            log(f"  [resume] already done — skipping")
            # Load existing result
            result_path = os.path.join(design_dir, "result.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    results.append(json.load(f))
            continue

        t0 = time.time()

        # Build prompt
        user_prompt = build_user_prompt(icl_examples, design["rtl"])

        # Save prompt for reference
        with open(os.path.join(design_dir, "prompt.txt"), "w") as f:
            f.write(f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_prompt}")

        # Call LLM with timeout
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.design_timeout)

        try:
            response = llm.generate(SYSTEM_PROMPT, user_prompt)
        except DesignTimeoutError:
            log(f"  [timeout] design timed out after {args.design_timeout}s")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            result = {
                "design_id": did,
                "module_name": design["module_name"],
                "status": "timeout",
                "total_generated": 0,
                "wall_time_s": time.time() - t0,
            }
            results.append(result)
            with open(os.path.join(design_dir, "result.json"), "w") as f:
                json.dump(result, f, indent=2)
            continue
        except Exception as e:
            log(f"  [error] LLM call failed: {e}")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            result = {
                "design_id": did,
                "module_name": design["module_name"],
                "status": "error",
                "error": str(e),
                "total_generated": 0,
                "wall_time_s": time.time() - t0,
            }
            results.append(result)
            with open(os.path.join(design_dir, "result.json"), "w") as f:
                json.dump(result, f, indent=2)
            continue
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        elapsed = time.time() - t0

        # Save raw response
        with open(os.path.join(design_dir, "llm_response.txt"), "w") as f:
            f.write(response)

        # Parse assertions
        assertions, cleaned = extract_assertions(response)
        clock = design.get("clock") or "clk"
        sva_text = wrap_assertions_sv(assertions, clock=clock)

        # Save SVA
        with open(sva_path, "w") as f:
            f.write(sva_text)

        log(f"  Generated {len(assertions)} assertions in {elapsed:.1f}s")

        result = {
            "design_id": did,
            "module_name": design["module_name"],
            "clock": design.get("clock"),
            "reset": design.get("reset"),
            "status": "ok",
            "total_generated": len(assertions),
            "ground_truth_count": design.get("ground_truth_count", 0),
            "wall_time_s": round(elapsed, 1),
        }
        results.append(result)
        with open(os.path.join(design_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)

    # Summary
    log(f"\n{'='*60}")
    log(f"Baseline complete: {len(results)} designs processed")
    total_gen = sum(r.get("total_generated", 0) for r in results)
    log(f"  Total assertions generated: {total_gen}")

    # Save run metadata
    metadata = {
        "model": args.model,
        "provider": args.provider,
        "k_shot": args.k_shot,
        "total_designs": len(results),
        "total_assertions_generated": total_gen,
        "results": results,
    }
    with open(os.path.join(args.debug_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AssertionBench baseline: k-shot ICL assertion generation",
    )
    parser.add_argument("--debug_dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--provider", choices=["ollama", "openrouter", "vllm", "hf"],
                        default="ollama")
    parser.add_argument("--model", default="qwen3.5:35b")
    parser.add_argument("--k_shot", type=int, default=5,
                        help="Number of ICL examples (default: 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max designs to process")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N designs")
    parser.add_argument("--design_timeout", type=int, default=600,
                        help="Max seconds per design (default: 600)")
    parser.add_argument("--design_ids", nargs="+", default=None,
                        help="Only process these design IDs")
    parser.add_argument("--resume", action="store_true",
                        help="Skip designs with existing sva_assertion.sv")
    args = parser.parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
