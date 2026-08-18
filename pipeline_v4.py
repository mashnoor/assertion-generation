# HPC sqlite3 compatibility fix — must be at the very top before any chromadb import
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
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests as _http_client
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Path setup — repo root (contains designs.csv / assertion_specs.csv)
# ---------------------------------------------------------------------------

_CURSOR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURSOR_DIR not in sys.path:
    sys.path.insert(0, _CURSOR_DIR)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(*args, **kwargs):
    """Timestamped print for all pipeline output."""
    print(f"[{_ts()}]", *args, **kwargs)


# ---------------------------------------------------------------------------
# OllamaDirectEmbeddings — minimal requests-based Ollama embeddings
# (copied from pipeline_v3.py to avoid langchain_community import hang on HPC)
# ---------------------------------------------------------------------------

class OllamaDirectEmbeddings:
    """Minimal Ollama embeddings via requests. Implements langchain Embeddings interface.

    Replaces langchain_community.embeddings.OllamaEmbeddings to avoid import hangs.
    """
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = _http_client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# MinimalLLM — requests-based LLM (Ollama native + OpenAI-compat)
# ---------------------------------------------------------------------------

class _LLMResponse:
    def __init__(self, content: str):
        self.content = content


class MinimalLLM:
    """Minimal requests-based LLM supporting Ollama native API and OpenAI-compatible APIs.

    Replaces langchain_openai.ChatOpenAI to avoid openai/httpx import hangs on HPC.
    """
    def __init__(self, base_url: str, api_key: str = "", model: str = "",
                 temperature: float = 0.1, timeout: int = 420,
                 native_ollama: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.native_ollama = native_ollama

    def _msgs_to_dicts(self, messages):
        result = []
        for m in messages:
            if isinstance(m, dict):
                result.append(m)
            elif hasattr(m, "type"):
                role = {"ai": "assistant", "system": "system"}.get(m.type, "user")
                result.append({"role": role, "content": m.content})
            else:
                result.append({"role": "user", "content": str(m)})
        return result

    def _call(self, messages, structured_schema=None) -> str:
        msg_dicts = self._msgs_to_dicts(messages)
        if self.native_ollama:
            body = {
                "model": self.model,
                "messages": msg_dicts,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            if structured_schema is not None:
                body["format"] = structured_schema.model_json_schema()
            headers = {"Content-Type": "application/json"}
            endpoint = f"{self.base_url}/api/chat"
            resp = _http_client.post(endpoint, json=body, headers=headers,
                                     timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        else:
            body = {
                "model": self.model,
                "messages": msg_dicts,
                "stream": False,
                "temperature": self.temperature,
            }
            if structured_schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured_schema.__name__,
                        "schema": structured_schema.model_json_schema(),
                    },
                }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            endpoint = f"{self.base_url}/chat/completions"
            resp = _http_client.post(endpoint, json=body, headers=headers,
                                     timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def invoke(self, messages) -> _LLMResponse:
        return _LLMResponse(self._call(messages))

    def chat_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """Call Ollama native /api/chat with tools parameter."""
        if not self.native_ollama:
            raise RuntimeError("chat_with_tools requires native_ollama=True")
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        headers = {"Content-Type": "application/json"}
        endpoint = f"{self.base_url}/api/chat"
        resp = _http_client.post(endpoint, json=body, headers=headers,
                                 timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]


# ---------------------------------------------------------------------------
# HFTransformersLLM — local HuggingFace model inference
# ---------------------------------------------------------------------------

class HFTransformersLLM:
    """LLM using HuggingFace transformers for local inference.

    Same interface as MinimalLLM (invoke, _call, chat_with_tools stub).
    Qwen3.5 is multimodal (Qwen3_5ForConditionalGeneration) — uses
    AutoProcessor + flash-linear-attention for fast Gated DeltaNet kernels.
    """
    def __init__(self, model_name: str, temperature: float = 0.1,
                 max_new_tokens: int = 4096, device: str = "auto"):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        log(f"Loading HF model: {model_name} (this may take a few minutes)...")
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

    def _msgs_to_dicts(self, messages):
        """Convert LangChain-style messages to plain dicts with string content."""
        result = []
        for m in messages:
            if isinstance(m, dict):
                content = m.get("content", "")
                role = m.get("role", "user")
            elif hasattr(m, "type"):
                role = {"ai": "assistant", "system": "system"}.get(m.type, "user")
                content = m.content
            else:
                role = "user"
                content = str(m)
            if isinstance(content, list):
                content = " ".join(c.get("text", str(c)) for c in content)
            result.append({"role": role, "content": content})
        return result

    def _call(self, messages, structured_schema=None) -> str:
        import torch
        msg_dicts = self._msgs_to_dicts(messages)
        text = self.processor.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True,
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

    def invoke(self, messages) -> _LLMResponse:
        return _LLMResponse(self._call(messages))

    def chat_with_tools(self, messages, tools):
        raise RuntimeError("chat_with_tools not supported for HF provider")


# ---------------------------------------------------------------------------
# Embeddings factory
# ---------------------------------------------------------------------------

def _init_embeddings(provider: str):
    """Return an embeddings object appropriate for the current environment.

    Checks OLLAMA_EMBEDDING_MODEL first (set on HPC); falls back to
    HuggingFaceEmbeddings(all-MiniLM-L6-v2) for local workstation use.
    """
    ollama_emb = os.getenv("OLLAMA_EMBEDDING_MODEL")
    if ollama_emb:
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        log(f"Using OllamaDirectEmbeddings: model={ollama_emb} base={ollama_base}")
        return OllamaDirectEmbeddings(model=ollama_emb, base_url=ollama_base)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        log("Using HuggingFaceEmbeddings: all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _init_llm(provider: str, model: str) -> MinimalLLM:
    """Construct a MinimalLLM for the given provider."""
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        log(f"LLM provider=ollama model={model} base_url={base_url}")
        return MinimalLLM(
            base_url=base_url,
            model=model,
            temperature=0.1,
            timeout=480,
            native_ollama=True,
        )
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        log(f"LLM provider=openrouter model={model} base_url={base_url}")
        return MinimalLLM(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.1,
            timeout=480,
            native_ollama=False,
        )
    elif provider == "vllm":
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        log(f"LLM provider=vllm model={model} base_url={base_url}")
        return MinimalLLM(
            base_url=base_url,
            api_key="dummy",
            model=model,
            temperature=0.1,
            timeout=480,
            native_ollama=False,
        )
    elif provider == "hf":
        log(f"LLM provider=hf model={model}")
        return HFTransformersLLM(
            model_name=model,
            temperature=0.1,
            max_new_tokens=4096,
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'ollama', 'openrouter', 'vllm', or 'hf'.")


# ---------------------------------------------------------------------------
# Design complexity
# ---------------------------------------------------------------------------

def compute_module_count(rtl_code: str) -> int:
    """Count module declarations via regex."""
    return len(re.findall(r"^\s*module\s+\w+", rtl_code, re.MULTILINE))


# ---------------------------------------------------------------------------
# Git-based checkpointer
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = "checkpoint.json"


class GitCheckpointer:
    """Manages checkpoint.json for pipeline progress tracking.

    Checkpoint is stored inside the debug_dir (per-run) so different runs
    don't interfere with each other.
    """

    def __init__(self, base_dir: str, debug_dir: str = ""):
        """
        Args:
            base_dir: Absolute path to the cursor_style/ directory.
            debug_dir: Per-run output directory — checkpoint stored here.
        """
        self.base_dir = base_dir
        # Store checkpoint in debug_dir (per-run) if available, else base_dir
        ckpt_dir = debug_dir if debug_dir else base_dir
        self._checkpoint_path = os.path.join(ckpt_dir, CHECKPOINT_FILE)
        self._data = self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict:
        if os.path.exists(self._checkpoint_path):
            try:
                with open(self._checkpoint_path) as f:
                    return json.load(f)
            except Exception as e:
                log(f"[checkpoint] Warning: could not load checkpoint: {e}")
        return {
            "completed_designs": {},
            "skipped_designs": {},
            "total_specs_done": 0,
        }

    def _save(self):
        try:
            with open(self._checkpoint_path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            log(f"[checkpoint] Warning: could not save checkpoint: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_complete(self, design_id: str) -> bool:
        """Return True if design_id is recorded as completed."""
        return design_id in self._data.get("completed_designs", {})

    def is_skipped(self, design_id: str) -> bool:
        """Return True if design_id is recorded as skipped."""
        return design_id in self._data.get("skipped_designs", {})

    def mark_design_complete(self, design_id: str, metrics: Dict):
        """Record design completion and git-commit progress."""
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            **metrics,
        }
        self._data.setdefault("completed_designs", {})[design_id] = entry
        specs_done = metrics.get("specs_done", 0)
        self._data["total_specs_done"] = (
            self._data.get("total_specs_done", 0) + specs_done
        )
        self._save()
        self.git_commit(f"checkpoint: design {design_id} complete "
                        f"({specs_done} specs, avg_func={metrics.get('avg_func', 0):.3f})")

    def mark_design_skipped(self, design_id: str, reason: str):
        """Record a skipped design (e.g. timeout)."""
        self._data.setdefault("skipped_designs", {})[design_id] = {
            "reason": reason,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        self._save()
        self.git_commit(f"checkpoint: design {design_id} skipped ({reason})")

    def git_commit(self, message: str):
        """Stage results/ and checkpoint.json in base_dir and commit."""
        try:
            subprocess.run(
                ["git", "add", CHECKPOINT_FILE, "results/"],
                cwd=self.base_dir,
                capture_output=True,
                check=False,
                timeout=30,
            )
            result = subprocess.run(
                ["git", "commit", "-m", message,
                 "--author=pipeline_v4 <pipeline@hpc>"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            if result.returncode == 0:
                log(f"[git] committed: {message}")
            else:
                # Nothing to commit is normal when output dir is outside base_dir
                stderr = result.stderr.strip()
                if "nothing to commit" not in stderr and "nothing added" not in stderr:
                    log(f"[git] commit failed (rc={result.returncode}): {stderr[:200]}")
        except Exception as e:
            log(f"[git] commit error: {e}")

    # ------------------------------------------------------------------
    # Summary accessors
    # ------------------------------------------------------------------

    @property
    def completed_designs(self) -> Dict:
        return self._data.get("completed_designs", {})

    @property
    def skipped_designs(self) -> Dict:
        return self._data.get("skipped_designs", {})

    @property
    def total_specs_done(self) -> int:
        return self._data.get("total_specs_done", 0)


# ---------------------------------------------------------------------------
# Per-spec completion check
# ---------------------------------------------------------------------------

def _spec_is_done(spec_dir: str) -> bool:
    """Return True if sva_assertion.sv already exists (resume guard)."""
    return os.path.exists(os.path.join(spec_dir, "sva_assertion.sv"))


# ---------------------------------------------------------------------------
# Design RTL persistence helpers
# ---------------------------------------------------------------------------

def _save_module_rtl(design_dir: str, rtl_code: str, design_id: str = "design"):
    """Write all module RTLs to design_dir/<module_name>/rtl.sv for DesignTools."""
    # Save top-level `define lines for JG compilation
    define_lines = [l for l in rtl_code.splitlines() if re.match(r'\s*`define\s+\w+\s+\d+', l)]
    if define_lines:
        with open(os.path.join(design_dir, "defines.sv"), "w") as f:
            f.write("\n".join(define_lines) + "\n")
    try:
        import hierarchy as _hier
        hierarchy_obj = _hier.decompose_design(design_id, rtl_code)
        for mod_name, mod_info in hierarchy_obj.modules.items():
            mod_dir = os.path.join(design_dir, mod_name)
            os.makedirs(mod_dir, exist_ok=True)
            rtl_path = os.path.join(mod_dir, "rtl.sv")
            with open(rtl_path, "w") as f:
                f.write(mod_info.code)
        # Save design_graph.json for DesignTools.get_hierarchy() and _top_module_name()
        graph = {
            "sorted_modules": list(hierarchy_obj.sorted_modules),
            "adjacency_list": {k: list(v) for k, v in hierarchy_obj.adjacency_list.items()},
        }
        graph_path = os.path.join(design_dir, "design_graph.json")
        with open(graph_path, "w") as f:
            json.dump(graph, f, indent=2)
        return graph["sorted_modules"]
    except Exception as e:
        log(f"  [warn] hierarchy decomposition failed: {e}; writing flat rtl.sv")
        # Fallback: write whole RTL under a single pseudo-module
        mod_dir = os.path.join(design_dir, "_flat")
        os.makedirs(mod_dir, exist_ok=True)
        with open(os.path.join(mod_dir, "rtl.sv"), "w") as f:
            f.write(rtl_code)
        # Extract module names via regex so design_graph.json is always written
        module_names = re.findall(r'\bmodule\s+(\w+)', rtl_code)
        graph = {"sorted_modules": module_names, "adjacency_list": {}}
        with open(os.path.join(design_dir, "design_graph.json"), "w") as f:
            json.dump(graph, f, indent=2)
        return module_names


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _func_score(result) -> float:
    """Compute functionality score from SVAResult."""
    total = result.total if result.total > 0 else 1
    return result.proven / total


def _compute_design_metrics(results) -> Dict:
    """Aggregate per-spec SVAResult list into a summary metrics dict."""
    if not results:
        return {"specs_done": 0, "avg_func": 0.0, "avg_syntax": 0.0,
                "proven": 0, "falsified": 0, "undetermined": 0}
    syntax_scores = []
    func_scores = []
    proven_total = falsified_total = undetermined_total = 0
    for r in results:
        syntax_ok = 1.0 if r.status not in ("syntax_error", "compilation_error") else 0.0
        syntax_scores.append(syntax_ok)
        func_scores.append(_func_score(r))
        proven_total += r.proven
        falsified_total += r.falsified
        undetermined_total += r.undetermined
    return {
        "specs_done": len(results),
        "avg_syntax": sum(syntax_scores) / len(syntax_scores),
        "avg_func": sum(func_scores) / len(func_scores),
        "proven": proven_total,
        "falsified": falsified_total,
        "undetermined": undetermined_total,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """Main entry point: load data, sort designs, run agent per spec, checkpoint."""

    # ------------------------------------------------------------------
    # 1. Load CSV data
    # ------------------------------------------------------------------
    log(f"Loading designs from: {args.designs_csv}")
    designs_df = pd.read_csv(args.designs_csv)
    log(f"Loading specs from:   {args.specs_csv}")
    specs_df = pd.read_csv(args.specs_csv)

    # Normalise column names (strip whitespace)
    designs_df.columns = [c.strip() for c in designs_df.columns]
    specs_df.columns   = [c.strip() for c in specs_df.columns]

    # Build a dict: design_id -> (rtl_code, design_type)
    design_map: Dict[str, Tuple[str, str]] = {}
    for _, row in designs_df.iterrows():
        did   = str(row["id"]).strip()
        rtl   = str(row.get("rtl", "")).strip()
        dtype = str(row.get("type", "pipeline")).strip()
        design_map[did] = (rtl, dtype)

    # Build a dict: design_id -> list of (spec_id, spec_text)
    spec_map: Dict[str, List[Tuple[str, str]]] = {}
    for _, row in specs_df.iterrows():
        spec_id   = str(row["id"]).strip()
        spec_text = str(row["spec"]).strip()
        parent    = str(row.get("parent_design_id", "")).strip()
        if parent not in spec_map:
            spec_map[parent] = []
        spec_map[parent].append((spec_id, spec_text))

    # ------------------------------------------------------------------
    # 2. Sort designs by complexity (module count ascending, then RTL len)
    # ------------------------------------------------------------------
    design_list = []
    for did, (rtl, dtype) in design_map.items():
        mc = compute_module_count(rtl)
        design_list.append((did, rtl, dtype, mc))

    design_list.sort(key=lambda x: (x[3], len(x[1])))

    # Apply --design_ids filter (overrides --limit/--offset)
    if args.design_ids:
        id_set = set(args.design_ids)
        design_list = [d for d in design_list if d[0] in id_set]
    else:
        offset = getattr(args, 'offset', 0) or 0
        if offset:
            design_list = design_list[offset:]
        if args.limit:
            design_list = design_list[:args.limit]

    log(f"\nComplexity-sorted design order ({len(design_list)} designs):")
    for rank, (did, rtl, dtype, mc) in enumerate(design_list, 1):
        log(f"  {rank:3d}. {did}  modules={mc}  rtl_chars={len(rtl)}")

    # ------------------------------------------------------------------
    # 3. Initialise shared resources
    # ------------------------------------------------------------------
    os.makedirs(args.debug_dir, exist_ok=True)

    log("\nInitialising embeddings...")
    embeddings = _init_embeddings(args.provider)

    log("Initialising ChromaDB vector store...")
    vector_store = Chroma(
        collection_name="rtl_ast_chunks",
        embedding_function=embeddings,
        persist_directory=args.db_path,
    )

    log("Initialising LLM...")
    llm = _init_llm(args.provider, args.model)

    checkpointer = GitCheckpointer(base_dir=_CURSOR_DIR, debug_dir=args.debug_dir)

    # Import agent components after path setup
    from ast_indexer import ASTIndexer
    from tools import DesignTools
    from agent import SVAAgent, SVAResult

    indexer = ASTIndexer(
        db_path=args.db_path,
        embeddings=embeddings,
        collection_name="rtl_ast_chunks",
    )

    # ------------------------------------------------------------------
    # 4. Per-design processing loop
    # ------------------------------------------------------------------
    pipeline_start = time.time()
    all_results: List[Dict] = []   # flat list for run_metadata
    designs_completed = 0
    designs_skipped = 0

    for design_idx, (design_id, rtl_code, design_type, module_count) in enumerate(design_list):
        log(f"\n{'='*70}")
        log(f"Design {design_idx+1}/{len(design_list)}: {design_id}"
            f"  type={design_type}  modules={module_count}")

        # ---- Resume check ----
        if args.resume and checkpointer.is_complete(design_id):
            log(f"  [resume] design {design_id} already complete — skipping")
            designs_completed += 1
            continue
        if checkpointer.is_skipped(design_id):
            log(f"  [resume] design {design_id} was previously skipped — skipping")
            designs_skipped += 1
            continue

        design_start = time.time()
        design_dir   = os.path.join(args.debug_dir, design_id)
        os.makedirs(design_dir, exist_ok=True)

        # ---- Phase 0: AST indexing ----
        if not args.no_index:
            if indexer.is_design_indexed(design_id):
                log(f"  [phase0] design {design_id} already indexed — skipping index")
            else:
                log(f"  [phase0] indexing design {design_id}...")
                try:
                    t0 = time.time()
                    n_chunks = indexer.index_design(design_id, rtl_code, design_type)
                    log(f"  [phase0] indexed {n_chunks} chunks in {time.time()-t0:.1f}s")
                except Exception as e:
                    log(f"  [phase0] ERROR indexing design {design_id}: {e}")
                    # Continue — tools that need ChromaDB will fall back gracefully
        else:
            log(f"  [phase0] --no_index set — skipping indexing")

        # ---- Persist module RTLs to design_dir for DesignTools ----
        sorted_mods = _save_module_rtl(design_dir, rtl_code, design_id)
        top_module  = sorted_mods[0] if sorted_mods else design_id

        # ---- Collect specs for this design ----
        specs = spec_map.get(design_id, [])
        if args.spec_limit:
            specs = specs[:args.spec_limit]

        if not specs:
            log(f"  [warn] no specs found for design {design_id}")

        # ---- Per-spec agent loop ----
        design_results: List = []
        timed_out = False

        for spec_idx, (spec_id, spec_text) in enumerate(specs):
            elapsed_design = time.time() - design_start
            if elapsed_design >= args.design_timeout:
                log(f"  [timeout] design wall time {elapsed_design:.0f}s >= "
                    f"{args.design_timeout}s — stopping early")
                timed_out = True
                break

            spec_dir = os.path.join(design_dir, "specs", spec_id)
            os.makedirs(spec_dir, exist_ok=True)

            # Resume: skip if already done
            if args.resume and _spec_is_done(spec_dir):
                log(f"  [resume] spec {spec_id} already done — skipping")
                # Load prior result for metrics
                result_path = os.path.join(spec_dir, "result.json")
                if os.path.exists(result_path):
                    try:
                        with open(result_path) as f:
                            prior = json.load(f)
                        # Reconstruct minimal SVAResult for metrics
                        from dataclasses import fields
                        dummy = SVAResult(
                            spec_id=spec_id,
                            sva_code="",
                            status=prior.get("status", "unknown"),
                            proven=prior.get("proven", 0),
                            falsified=prior.get("falsified", 0),
                            undetermined=prior.get("undetermined", 0),
                            total=prior.get("total", 0),
                            vacuity_status=prior.get("vacuity_status", "not_checked"),
                            tool_calls=[],
                            jg_iterations=prior.get("jg_iterations", 0),
                            wall_time_s=prior.get("wall_time_s", 0.0),
                            error_message=prior.get("error_message", ""),
                            context_rounds=prior.get("context_rounds", 0),
                        )
                        design_results.append(dummy)
                    except Exception:
                        pass
                continue

            # Write spec text for reference
            with open(os.path.join(spec_dir, "spec.txt"), "w") as f:
                f.write(spec_text)

            log(f"  [spec {spec_idx+1}/{len(specs)}] {spec_id}: {spec_text[:80]}...")

            # Remaining budget for this spec
            remaining = args.design_timeout - (time.time() - design_start)
            spec_timeout = min(args.spec_timeout, int(remaining) - 10)
            if spec_timeout <= 0:
                log(f"  [timeout] no time budget left for spec {spec_id}")
                timed_out = True
                break

            # Construct per-spec tools + agent
            ablation = getattr(args, 'ablation', 'full')
            tools_obj = DesignTools(
                vector_store=vector_store,
                design_id=design_id,
                design_dir=design_dir,
                llm=llm,
            )
            # Ablation: control which tools and phases are available
            max_ctx = 6
            max_ver = 3
            disabled_tools = set()
            if ablation == "no_verify":
                max_ver = 0  # skip Phase B entirely
            elif ablation == "no_jg_tools":
                disabled_tools = {"get_flop_info", "get_fanin", "get_fanout",
                                  "verify_sva", "check_vacuity"}
                max_ver = 0  # can't verify without JG
            elif ablation == "no_rag":
                disabled_tools = {"search_design", "get_module_info",
                                  "get_hierarchy", "get_parameters",
                                  "get_always_blocks"}
            elif ablation == "no_tools":
                max_ctx = 0  # no context gathering at all
                max_ver = 0  # no verification
            agent = SVAAgent(
                llm=llm,
                tools=tools_obj,
                spec_id=spec_id,
                debug_dir=spec_dir,
                max_context_rounds=max_ctx,
                max_verify_rounds=max_ver,
                disabled_tools=disabled_tools,
            )

            t_spec = time.time()
            try:
                result = agent.run(
                    spec_text=spec_text,
                    design_id=design_id,
                    top_module=top_module,
                    timeout_s=spec_timeout,
                )
            except Exception as e:
                log(f"  [error] agent.run raised: {e}")
                result = SVAResult(
                    spec_id=spec_id,
                    sva_code="",
                    status="unknown",
                    proven=0, falsified=0, undetermined=0, total=0,
                    vacuity_status="not_checked",
                    tool_calls=[],
                    jg_iterations=0,
                    wall_time_s=time.time() - t_spec,
                    error_message=str(e),
                    context_rounds=0,
                )

            spec_elapsed = time.time() - t_spec
            n_tools = len(result.tool_calls)
            log(f"  [TIMING] spec {spec_id} total={spec_elapsed:.1f}s "
                f"status={result.status} jg_iter={result.jg_iterations} tools={n_tools}")

            design_results.append(result)
            all_results.append({
                "design_id": design_id,
                "spec_id": spec_id,
                "status": result.status,
                "proven": result.proven,
                "falsified": result.falsified,
                "undetermined": result.undetermined,
                "total": result.total,
                "wall_time_s": result.wall_time_s,
            })

            # Persist result JSON
            result_path = os.path.join(spec_dir, "result.json")
            try:
                with open(result_path, "w") as f:
                    json.dump({
                        "spec_id":        result.spec_id,
                        "status":         result.status,
                        "proven":         result.proven,
                        "falsified":      result.falsified,
                        "undetermined":   result.undetermined,
                        "total":          result.total,
                        "vacuity_status": result.vacuity_status,
                        "jg_iterations":  result.jg_iterations,
                        "context_rounds": result.context_rounds,
                        "wall_time_s":    result.wall_time_s,
                        "error_message":  result.error_message,
                    }, f, indent=2)
            except Exception as e:
                log(f"  [warn] could not write result.json for {spec_id}: {e}")

        # ---- Design-level timing + metrics ----
        design_elapsed = time.time() - design_start
        metrics = _compute_design_metrics(design_results)
        log(f"[TIMING] design {design_id} total={design_elapsed:.1f}s "
            f"specs={metrics['specs_done']} avg_func={metrics['avg_func']:.3f}")

        # ---- Checkpoint ----
        if timed_out and not design_results:
            checkpointer.mark_design_skipped(design_id, reason="timeout")
            designs_skipped += 1
        else:
            checkpointer.mark_design_complete(design_id, metrics)
            designs_completed += 1

    # ------------------------------------------------------------------
    # 5. Write run_metadata.json
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    total_specs = len(all_results)

    avg_syntax = 0.0
    avg_func   = 0.0
    if all_results:
        syntax_vals = [1.0 if r["status"] not in ("syntax_error", "compilation_error")
                       else 0.0 for r in all_results]
        func_vals   = [r["proven"] / max(r["total"], 1) for r in all_results]
        avg_syntax  = sum(syntax_vals) / len(syntax_vals)
        avg_func    = sum(func_vals)   / len(func_vals)

    run_metadata = {
        "model":              args.model,
        "provider":           args.provider,
        "ablation":           getattr(args, 'ablation', 'full'),
        "total_designs":      len(design_list),
        "completed_designs":  designs_completed,
        "skipped_designs":    designs_skipped,
        "total_specs":        total_specs,
        "total_time_s":       round(total_time, 1),
        "avg_syntax":         round(avg_syntax, 4),
        "avg_func":           round(avg_func, 4),
    }

    metadata_path = os.path.join(args.debug_dir, "run_metadata.json")
    try:
        with open(metadata_path, "w") as f:
            json.dump(run_metadata, f, indent=2)
        log(f"\nRun metadata saved to: {metadata_path}")
    except Exception as e:
        log(f"[warn] could not write run_metadata.json: {e}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    log(f"\n{'='*70}")
    log(f"Pipeline complete in {total_time/60:.1f} min")
    log(f"  Designs processed : {len(design_list)}")
    log(f"  Completed         : {designs_completed}")
    log(f"  Skipped (timeout) : {designs_skipped}")
    log(f"  Total specs       : {total_specs}")
    log(f"  avg_syntax        : {avg_syntax:.3f}")
    log(f"  avg_func          : {avg_func:.3f}")
    log(f"  Checkpoint        : {checkpointer._checkpoint_path}")
    log(f"  Output dir        : {args.debug_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Cursor-style RTL SVA generation pipeline (pipeline_v4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--designs_csv",
        default=os.path.join(_CURSOR_DIR, "designs.csv"),
        help="Path to designs.csv (columns: id, rtl, type).",
    )
    parser.add_argument(
        "--specs_csv",
        default=os.path.join(_CURSOR_DIR, "assertion_specs.csv"),
        help="Path to assertion_specs.csv (columns: id, spec, parent_design_id).",
    )
    parser.add_argument(
        "--debug_dir",
        required=True,
        help="Output directory for all per-design/spec artifacts.",
    )
    parser.add_argument(
        "--db_path",
        default=os.path.join(_CURSOR_DIR, "chroma_db_cursor"),
        help="ChromaDB persist directory.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openrouter", "vllm", "hf"],
        default="ollama",
        help="LLM provider.",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:35b",
        help="Model name for the chosen provider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of designs to process (after complexity sort).",
    )
    parser.add_argument(
        "--spec_limit",
        type=int,
        default=None,
        help="Maximum specs to process per design.",
    )
    parser.add_argument(
        "--design_timeout",
        type=int,
        default=3600,
        help="Maximum wall-clock seconds per design before skipping (default: 3600).",
    )
    parser.add_argument(
        "--spec_timeout",
        type=int,
        default=480,
        help="Maximum wall-clock seconds per spec agent run (default: 480).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip designs/specs that are already complete (sva_assertion.sv exists).",
    )
    parser.add_argument(
        "--no_index",
        action="store_true",
        help="Skip Phase 0 AST indexing (assume ChromaDB already populated).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N designs after complexity sort (for SLURM array parallelism).",
    )
    parser.add_argument(
        "--design_ids",
        nargs="+",
        default=None,
        help="Only process these design IDs (space-separated). Overrides --limit.",
    )
    parser.add_argument(
        "--ablation",
        choices=["full", "no_verify", "no_jg_tools", "no_rag", "no_tools"],
        default="full",
        help=(
            "Ablation study mode. full=all features (default). "
            "no_verify=skip Phase B verification loop. "
            "no_jg_tools=disable JasperGold tools (keep RAG). "
            "no_rag=disable ChromaDB/RAG tools (keep JG). "
            "no_tools=no tool calls, just enhanced prompt."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    log(f"pipeline_v4 starting — provider={args.provider} model={args.model}")
    log(f"  designs_csv    : {args.designs_csv}")
    log(f"  specs_csv      : {args.specs_csv}")
    log(f"  debug_dir      : {args.debug_dir}")
    log(f"  db_path        : {args.db_path}")
    log(f"  design_timeout : {args.design_timeout}s")
    log(f"  spec_timeout   : {args.spec_timeout}s")
    log(f"  resume         : {args.resume}")
    log(f"  no_index       : {args.no_index}")
    run_pipeline(args)
