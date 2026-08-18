# Agentic SVA Generation Pipeline (cursor-style v4)

## Project Overview

Research project for automated **RTL formal verification assertion generation** using an agentic, tool-augmented LLM pipeline. A ReAct-style agent autonomously decides which tools to invoke (ChromaDB semantic search + JasperGold formal analysis) to gather design context, then generates SystemVerilog Assertions (SVA) with iterative formal verification feedback.

**Key result**: Pipeline achieves **0.937 syntax / 0.820 functionality** vs baseline **0.783 / 0.432** on 192 designs (576 specs) using qwen3.5:35b.

## Repository Structure

```
assertion_generation_cursor_style/
├── pipeline_v4.py               # Main orchestrator: AST indexing → ChromaDB → agent per spec
├── agent.py                     # ReAct agent: Phase A (context gathering) + Phase B (generation + verify)
├── tools.py                     # 10 tools: 5 ChromaDB + 5 JasperGold SSH
├── ast_indexer.py               # pyslang-based RTL AST chunker → ChromaDB documents
├── baseline_v4.py               # Baseline: raw RTL + spec → LLM, no tools/RAG
├── evaluate_v4.py               # JasperGold prove + syntax/functionality/vacuity metrics
├── export_results.py            # Paper-ready CSV/JSON/LaTeX from evaluation results
├── run_fveval_design2sva.py     # Run our pipeline on FVEval Design2SVA (RTL-only, no NL spec)
├── test_6_designs.csv           # 6-design subset for quick testing
├── designs.csv                  # 192 RTL designs
├── assertion_specs.csv          # 576 NL assertion specs
├── build_sva_examples.py        # Builds `sva_examples` ChromaDB collection from FVEval data
├── tests/                       # Test suite
├── for_assertion_bench/         # AssertionBench evaluation scripts (separate benchmark)
├── results/                     # All experiment outputs (gitignored)
│   ├── full_pipeline/           # qwen3.5:35b full pipeline (192 designs, 576 SVAs)
│   ├── full_baseline/           # qwen3.5:35b baseline (192 designs, 576 SVAs)
│   ├── mistral-small3_1_24b_*/  # Mistral Small 3.1 pipeline + baseline
│   ├── mixtral_8x7b_*/          # Mixtral 8x7B pipeline + baseline + ablations
│   ├── llama3_1_8b_*/           # LLaMA 3.1 8B pipeline + baseline + ablations
│   ├── llama3_1_70b_*/          # LLaMA 3.1 70B pipeline + baseline (partial)
│   ├── ablation_no_*/           # qwen3.5:35b ablation studies (36 designs each)
│   ├── ablation_comparison/     # Cross-ablation comparison CSVs
│   ├── fveval_*_design2sva_*/   # FVEval D2SVA benchmark runs (96 pipeline + 96 FSM, 5 trials each)
│   └── paper_results/           # Exported tables/CSVs/LaTeX for the research paper
├── rebuild_fveval_csvs.py       # Rebuild FVEval CSVs from trial dirs + compute pass@k
├── submit_fveval.sh             # Submit 12 individual FVEval SLURM jobs
├── slurm_fveval_single.sh       # Single parametrized SLURM job for FVEval
├── slurm_fveval_d2sva.sh        # SLURM array: 12 tasks for FVEval D2SVA benchmark
├── slurm_v4_full192.sh          # SLURM array: 32 nodes × 6 designs (primary)
├── slurm_v4_multimodel.sh       # SLURM multi-model experiment
├── slurm_v4_multimodel_eval.sh  # SLURM multi-model evaluation
├── slurm_v4_ablation_36.sh      # SLURM ablation study (36 designs)
├── slurm_v4_ablation.sh         # SLURM ablation (original 6 designs)
├── slurm_v4_eval.sh             # SLURM evaluation-only
├── slurm_v4_local_test6.sh      # SLURM 6-design test
├── slurm_v4.sh                  # SLURM single-node full run
├── slurm_v4_test5.sh            # SLURM 5-design test
├── run_all_models.sh            # Local multi-model runner
├── check_v4_progress.sh         # Progress monitor
├── pytest.ini                   # Pytest config
└── checkpoint.json              # Pipeline checkpoint state
```

## Architecture

### Two-Phase ReAct Agent (`agent.py`)

**Phase A — Context Gathering** (up to 6 tool-call rounds):
The agent reasons about what it needs and autonomously calls tools to query the RTL design. It builds understanding of module hierarchy, signal behavior, parameters, and connectivity.

**Phase B — SVA Generation + Verification Loop** (up to 5 iterations):
The agent generates SVA code, then uses `verify_sva` to formally prove it with JasperGold. On failure (syntax errors, falsified properties), JG feedback is fed back for iterative correction.

### 10 Tools (`tools.py` — `DesignTools` class)

**ChromaDB tools (fast, milliseconds):**
1. `search_design(query, k)` — Semantic search over AST chunks
2. `get_module_info(module_name)` — Module interface (ports, parameters)
3. `get_hierarchy()` — Full module dependency tree
4. `get_parameters(module_name)` — Parameter/localparam values
5. `get_always_blocks(signal_name, module_name)` — Always blocks mentioning a signal

**JasperGold tools (authoritative, seconds, via SSH):**
6. `get_fanin(signal, module)` — Signal fan-in cone
7. `get_fanout(signal, module)` — Signal fan-out cone
8. `get_flop_info(signal, module)` — Flip-flop properties (clock, reset, data)
9. `verify_sva(sva_code)` — Formal prove with JasperGold
10. `check_vacuity(sva_code)` — Vacuity check (non-trivial proof verification)

### AST Indexing (`ast_indexer.py`)

Uses **pyslang** to parse RTL into semantic chunks stored in ChromaDB:
- `MODULE_INTERFACE` — ports, parameters per module
- `ALWAYS_BLOCK` — one per always/always_ff/always_comb block
- `INSTANCE` — one per sub-module instantiation
- `ASSIGN` — continuous assign groups (capped at 20/module)

Falls back to regex-based extraction if pyslang is unavailable.

### Pipeline Orchestrator (`pipeline_v4.py`)

1. Loads `designs.csv` + `assertion_specs.csv` from the repo root
2. For each design: runs AST indexer → populates per-task ChromaDB
3. For each spec: launches ReAct agent with `DesignTools`
4. Agent output saved as `sva_assertion.sv` + `agent_result.json` (tool log, timing)
5. Supports `--offset`, `--limit`, `--resume` for distributed SLURM runs

### Baseline (`baseline_v4.py`)

Sends raw RTL + NL spec directly to the LLM in a single prompt. No tools, no RAG, no verification loop. Output layout matches pipeline for apples-to-apples evaluation.

## HPC / Cluster Environment

### Locations
- **JasperGold**: configured via `JASPERGOLD_SSH_HOST` env var (SSH alias, key-based auth recommended)
- **GitHub**: `git@github.com:mashnoor/assertion-generation.git` branch `master`

### Quick Start
```bash
module load python/python-3.11.4-gcc-12.2.0   # adjust to your cluster's Python module
source "${VENV_PATH:-$HOME/venv}/bin/activate"
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1

# Run pipeline (6 designs, local test):
sbatch slurm_v4_local_test6.sh

# Run full 192-design experiment (32-node array):
sbatch slurm_v4_full192.sh

# Run multi-model experiment:
sbatch slurm_v4_multimodel.sh

# Run ablation study (36 designs):
sbatch slurm_v4_ablation_36.sh

# Evaluate after jobs complete:
sbatch --dependency=afterok:<job_id> slurm_v4_eval.sh

# Export paper results:
python export_results.py --results_dirs results/full_pipeline results/full_baseline \
    --labels "Pipeline" "Baseline" --designs_csv designs.csv --output_dir results/paper_results/

# FVEval Design2SVA benchmark (RTL-only, apples-to-apples with FVEval Table III):
sbatch slurm_fveval_d2sva.sh
# Then evaluate with FVEval's evaluator:
# cd FVEval && python run_evaluation.py -i ../results/fveval_<model>_design2sva_pipeline --task design2sva
```

### SLURM Array Design (`slurm_v4_full192.sh`)
- `--array=0-31` → 32 tasks × 6 designs = 192 total
- Each task starts its own Ollama instance on a unique port (`11434 + TASK_ID`)
- Task 0 builds ChromaDB `sva_examples`; others wait for sentinel `.hpc_chroma_ready`
- Per-task ChromaDB: `chroma_db_cursor_task_<N>` (isolated to avoid lock contention)
- `--constraint=h100`, `--gres=gpu:1`, `--mem=64G`

### Key Environment Variables
- `OLLAMA_BASE_URL` — Ollama endpoint (auto-set in SLURM scripts)
- `OLLAMA_EMBEDDING_MODEL=qwen3-embedding:latest` — Embedding model for ChromaDB on HPC
- `OPENBLAS_NUM_THREADS=1`, `RAYON_NUM_THREADS=1` — Required on login node

### HPC Notes
- **Venv**: override location with `VENV_PATH` (default `$HOME/venv`); needs `pysqlite3-binary`, `pyslang`
- **Ollama**: `~/bin/ollama`, models cached at `~/.ollama/models/`
- **SQLite3 fix**: `pysqlite3` override at top of Python files
- **Login node limits**: Cannot run pipeline directly (memory/thread limits). Use SLURM.

## Data Dependencies

Pipeline reads from the repo root:
- `designs.csv` — 192 RTL designs (`id`, `rtl`, `type`: pipeline|fsm)
- `assertion_specs.csv` — 576 NL specs (`id`, `spec`, `parent_design_id`)
- `FVEval/` — NVIDIA FVEval benchmark data, cloned into the repo root (for `sva_examples` collection)

## Evaluation (`evaluate_v4.py`)

Runs JasperGold prove on all `sva_assertion.sv` files. Metrics per spec:
- **syntax**: 1.0 if no compilation error, 0.0 otherwise
- **functionality**: `proven / total_properties`
- **func_relaxed**: `(proven + undetermined) / total`
- **vacuity**: Confirms non-trivial proofs (optional `--vacuity` flag)
- **tool_calls**, **jg_iterations**, **wall_time_s**: From `agent_result.json`

Output: `evaluation_results.csv` (per-spec) + `evaluation_summary.json` (aggregates)

## Experiment Results

Five experiment types total:
1. **Main Pipeline vs Baseline** — 192 designs, 576 specs (our dataset with NL specs)
2. **Multi-Model Comparison** — pipeline + baseline across 5 models
3. **Ablation Study** — 36 designs, 5 configurations (what each component contributes)
4. **FVEval Design2SVA** — external benchmark, RTL-only, 96+96 designs, pass@k
5. **AssertionBench** — external benchmark (for_assertion_bench/)

### 1. Main Results (qwen3.5:35b, 192 designs, 576 specs)

| Method | Syntax | Functionality | Proven | Falsified |
|--------|--------|---------------|--------|-----------|
| **Pipeline** | **0.937** | **0.820** | 790 | 260 |
| Baseline | 0.783 | 0.432 | 445 | 246 |

Pipeline wins on 127/192 designs (66.1%), ties 55 (28.6%), loses 10 (5.2%).

### 2. Multi-Model Comparison

| Model | Pipeline Func | Baseline Func | Improvement |
|-------|---------------|---------------|-------------|
| qwen3.5:35b | 0.820 | 0.432 | +89.7% |
| mistral-small3.1:24b | 0.533 | 0.311 | +71.4% |
| mixtral:8x7b | pending | pending | — |
| llama3.1:8b | pending | pending | — |

### 3. Ablation Study (qwen3.5:35b, 36 designs, 108 specs)

| Configuration | Syntax | Func | vs Full Pipeline |
|---------------|--------|------|-----------------|
| Full pipeline | 0.937 | 0.820 | — |
| No RAG | 0.909 | 0.709 | -14% |
| No verify loop | 0.829 | 0.647 | -21% |
| No JG tools | 0.800 | 0.584 | -29% |
| No tools at all | 0.703 | 0.361 | -56% |

### Design Complexity Breakdown

| Complexity | Pipeline Func | Baseline Func | Relative Gain |
|------------|---------------|---------------|---------------|
| Small (1 module) | 0.931 | 0.690 | +34.9% |
| Medium (3-6 modules) | 0.787 | 0.494 | +59.3% |
| Large (11-51 modules) | 0.644 | 0.254 | +153.5% |

Pipeline advantage grows with design complexity.

### 4. FVEval Design2SVA Comparison

Our 192 designs are the exact same designs from FVEval's Design2SVA benchmark
(96 pipeline + 96 FSM). `run_fveval_design2sva.py` runs our tool-augmented
pipeline on FVEval's exact task (RTL only, no NL spec, 1 assertion per trial)
for apples-to-apples comparison with FVEval Table III.

Submit via `./submit_fveval.sh` (12 individual jobs, 5 trials/design).
Rebuild CSVs and metrics: `python rebuild_fveval_csvs.py`.

**Comparison with FVEval Table III (Design2SVA):**

FVEval Table III reports pass@k for syntax and functional correctness.
Functional correctness = "whether the model has generated an assertion that can be proven."
Our metric: proven > 0 and falsified == 0 (aligned with FVEval's definition).

Pipeline designs (96):

| Method | Syntax@1 | Syntax@5 | Func@1 | Func@5 |
|--------|----------|----------|--------|--------|
| gpt-4o (FVEval) | 0.862 | 1.000 | 0.104 | 0.427 |
| gemini-1.5-pro (FVEval) | 0.665 | 1.000 | 0.175 | 0.500 |
| gemini-1.5-flash (FVEval) | 0.969 | 1.000 | 0.125 | 0.498 |
| **Ours (qwen3.5:35b + tools)** | **0.898** | — | **0.412** | **0.896** |

FSM designs (96):

| Method | Syntax@1 | Syntax@5 | Func@1 | Func@5 |
|--------|----------|----------|--------|--------|
| gpt-4o (FVEval) | 0.993 | 1.000 | 0.373 | 0.900 |
| gemini-1.5-pro (FVEval) | 0.556 | 1.000 | 0.427 | 0.906 |
| **Ours (qwen3.5:35b + tools)** | **0.979** | — | **0.860** | **0.958** |

Combined (96+96): Func@1=0.636, Func@5=0.927

Key findings:
- Pipeline Func@1: 0.412 vs gpt-4o 0.104 (~4x improvement)
- FSM Func@1: 0.860 vs gpt-4o 0.373 (~2.3x improvement)
- Our open-weight qwen3.5:35b + tool-augmented pipeline significantly outperforms
  GPT-4o and Gemini on the Design2SVA benchmark using the same evaluation criteria.

Note: FVEval Table I (NL2SVA-Human) and Table II/3 (NL2SVA-Machine) are different tasks
that include NL descriptions. Table III (Design2SVA) is RTL-only — our comparison target.

## Debug Output Structure

```
results/<run_name>/<design_id>/
├── specs/<spec_id>/
│   ├── sva_assertion.sv          # Final SVA output
│   ├── agent_result.json         # Tool log, timing, iterations
│   ├── eval_jg_output.txt        # JG prove output (from evaluate_v4)
│   ├── eval_metrics.json         # Per-spec metrics
│   └── eval_vacuity.txt          # Vacuity check output (if --vacuity)
├── <module_name>/
│   └── rtl.sv                    # Module RTL source
└── run_metadata.json             # Design-level metadata
```

## Common Commands

```bash
# Activate venv
source "${VENV_PATH:-$HOME/venv}/bin/activate"

# Run pipeline on 6 test designs (local, ollama on port 8080)
OLLAMA_BASE_URL=http://localhost:8080 python pipeline_v4.py \
    --provider ollama --model qwen3.5:35b \
    --debug_dir results/test --limit 6

# Run baseline
OLLAMA_BASE_URL=http://localhost:8080 python baseline_v4.py \
    --provider ollama --model qwen3.5:35b \
    --debug_dir results/test_baseline --limit 6

# Evaluate
python evaluate_v4.py --debug_dir results/test --designs_csv designs.csv

# Export paper results
python export_results.py \
    --results_dirs results/full_pipeline results/full_baseline \
    --labels "Pipeline" "Baseline" \
    --designs_csv designs.csv \
    --output_dir results/paper_results/

# Check progress
./check_v4_progress.sh
```

## Dependencies

- Python 3.11+
- `pandas`, `requests`, `pyslang` (AST parsing)
- `langchain-chroma`, `langchain-core`, `chromadb`
- `pysqlite3-binary` (HPC sqlite3 fix)
- Cadence JasperGold via SSH (`JASPERGOLD_SSH_HOST` env var)
- Ollama with `qwen3.5:35b` and `qwen3-embedding:latest`

## Key Design Decisions

- **No langchain LLM wrappers**: `MinimalLLM` uses raw `requests` to avoid import hangs on HPC (langchain_openai pulls httpx/openai which are slow to load).
- **No HuggingFace embeddings**: `OllamaDirectEmbeddings` avoids torch dependency and race conditions on shared HPC nodes.
- **Per-task ChromaDB**: Each SLURM task gets its own ChromaDB directory to avoid SQLite lock contention.
- **pyslang AST chunking**: More semantically meaningful than regex-based extraction. Regex fallback ensures robustness.
- **Native Ollama tool calling**: Agent uses Ollama's native `/api/chat` with `tools` parameter for function calling (no OpenAI compatibility layer needed).
