# ProofLoop

**Agentic, tool-augmented generation of SystemVerilog Assertions (SVA) for RTL formal verification.**

A ReAct-style LLM agent autonomously decides which tools to call — semantic search over an AST-chunked
RTL index, and live JasperGold formal queries — to gather design context, then generates SVA with an
iterative formal-verification feedback loop (JasperGold `verify_sva` / `check_vacuity`).

📄 Paper: **["From Language to Logic: Bridging LLMs & Formal Representations for RTL Assertion
Generation"](https://arxiv.org/abs/2604.23100)** — Nowfel Mashnoor, Hadi Kamali, Kimia Azar (arXiv:2604.23100)

**Headline result:** 93.7% syntax correctness / 82.0% functional correctness on a 192-design, 576-spec
benchmark, vs. 78.3% / 43.2% for a tool-free baseline on the same designs and LLM.

---

## Architecture

Two-phase ReAct agent (`agent.py`), driven per-assertion-spec by `pipeline_v4.py`:

- **Phase A — Context gathering** (up to 6 tool-call rounds): the agent reasons about what it needs
  to understand the design (module hierarchy, signal behavior, parameters, connectivity) and calls
  tools to find out, rather than being handed a fixed context window of RTL.
- **Phase B — Generation + verification loop** (up to 5 iterations): the agent drafts SVA, formally
  proves it with JasperGold via `verify_sva`, and on failure (syntax error or falsified property) feeds
  the JasperGold output back to itself for correction.

### Tools (`tools.py` — `DesignTools`)

| Backend | Tool | Purpose |
|---|---|---|
| ChromaDB (ms) | `search_design(query, k)` | Semantic search over AST chunks |
| ChromaDB | `get_module_info(module_name)` | Module ports/parameters |
| ChromaDB | `get_hierarchy()` | Full module dependency tree |
| ChromaDB | `get_parameters(module_name)` | Parameter/localparam values |
| ChromaDB | `get_always_blocks(signal_name, module_name)` | Always blocks touching a signal |
| JasperGold via SSH (s) | `get_fanin(signal, module)` | Signal fan-in cone |
| JasperGold | `get_fanout(signal, module)` | Signal fan-out cone |
| JasperGold | `get_flop_info(signal, module)` | Flip-flop clock/reset/data properties |
| JasperGold | `verify_sva(sva_code)` | Formally prove a candidate assertion |
| JasperGold | `check_vacuity(sva_code)` | Reject vacuously-true proofs |

### AST indexing (`ast_indexer.py`)

RTL is parsed with **pyslang** into semantic chunks (`MODULE_INTERFACE`, `ALWAYS_BLOCK`, `INSTANCE`,
`ASSIGN`) stored in a per-run ChromaDB collection. Falls back to regex-based extraction if pyslang is
unavailable.

### Baseline (`baseline_v4.py`)

Sends raw RTL + the natural-language spec directly to the LLM in a single prompt — no tools, no RAG,
no verification loop. Same output layout as the pipeline for apples-to-apples evaluation.

---

## Repository layout

```
.
├── pipeline_v4.py                # Orchestrator: AST index → ChromaDB → ReAct agent, per spec
├── agent.py                      # Phase A (context gathering) + Phase B (generate + verify)
├── tools.py                      # DesignTools: 5 ChromaDB tools + 5 JasperGold SSH tools
├── ast_indexer.py                # pyslang RTL → ChromaDB chunks
├── baseline_v4.py                # No-tools/no-RAG baseline
├── evaluate_v4.py                # JasperGold prove + syntax/functionality/vacuity metrics
├── export_results.py             # Aggregates evaluation output into paper-ready CSV/JSON/LaTeX
├── build_sva_examples.py         # Builds the `sva_examples` ChromaDB collection from FVEval NL2SVA data
├── run_fveval_design2sva.py      # Runs this pipeline on FVEval's Design2SVA task (RTL-only)
├── rebuild_fveval_csvs.py        # Rebuilds FVEval CSVs + pass@k from per-trial run dirs
├── designs.csv                   # 192 RTL designs (id, rtl, type: pipeline|fsm)
├── assertion_specs.csv           # 576 NL assertion specs (id, spec, parent_design_id)
├── test_6_designs.csv            # 6-design subset for smoke tests
├── slurm_v4*.sh                  # HPC job scripts for the main/multi-model/ablation experiments
├── slurm_fveval_*.sh, submit_fveval.sh  # HPC job scripts for the FVEval Design2SVA benchmark
├── for_assertion_bench/          # AssertionBench (external benchmark) pipeline + baseline + eval
│   └── results_summary/          # Aggregated AssertionBench numbers (tracked)
├── results/
│   ├── paper_results/            # Final CSV/JSON/LaTeX tables behind the paper (tracked)
│   └── ablation_comparison/      # Ablation-study comparison tables (tracked)
│   └── ...                       # Raw per-design run output (gitignored — see "Raw outputs" below)
└── tests/                        # pytest suite for ast_indexer.py and tools.py
```

### Raw outputs are not versioned

`results/*` (per-design/per-spec transcripts, SLURM `.out`/`.err` logs) and
`for_assertion_bench/results/*` are gitignored — they run into hundreds of MB per experiment. Only the
aggregated tables/summaries that the paper's tables and figures are built from are tracked
(`results/paper_results/`, `results/ablation_comparison/`, `for_assertion_bench/results_summary/`).
Re-running an experiment (below) regenerates the raw outputs locally; `export_results.py` /
`rebuild_fveval_csvs.py` turn them back into the tracked summary format.

---

## Setup

Requires **Python 3.11+**, **Ollama** (or another OpenAI/Ollama-compatible LLM endpoint) for the
generator model, and **Cadence JasperGold** reachable over SSH for the formal-verification tools.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Fetch the external benchmark data used to build the retrieval example collection and to run the
Design2SVA comparison (NVIDIA's [FVEval](https://github.com/NVlabs/FVEval), cloned as a subdirectory):

```bash
git clone https://github.com/NVlabs/FVEval.git
python build_sva_examples.py   # builds the `sva_examples` ChromaDB collection from FVEval/data_nl2sva
```

`tools.py` / `evaluate_v4.py` shell out to JasperGold over SSH. The target is configurable via
environment variables (key-based auth via an `~/.ssh/config` alias is recommended):

```bash
export JASPERGOLD_SSH_HOST=jaspergold            # SSH host/alias running JasperGold
export JASPERGOLD_REMOTE_DIR=~/proofloop_jg_work # scratch dir on that host (default)
# Password auth is only used if SSHPASS is set (requires sshpass); prefer SSH keys.
```

The `slurm_v4*.sh` / `slurm_fveval_*.sh` / `for_assertion_bench/slurm_*.sh` scripts assume a SLURM
cluster with GPU nodes and a user-installed [Ollama](https://ollama.com) at `~/bin/ollama`. All repo
paths resolve relative to the script location, so only the cluster-specific bits need adjusting before
submitting: the `module load` line, `VENV_PATH` (defaults to `~/venv`), and the `#SBATCH` resource
flags. The local commands below don't need any of that.

---

## Running it

### Smoke test (local, 6 designs)

```bash
ollama serve &   # or point OLLAMA_BASE_URL at an already-running instance
ollama pull qwen3.5:35b

python pipeline_v4.py --provider ollama --model qwen3.5:35b \
    --designs_csv test_6_designs.csv --specs_csv assertion_specs.csv \
    --debug_dir results/test --limit 6

python baseline_v4.py --provider ollama --model qwen3.5:35b \
    --designs_csv test_6_designs.csv --specs_csv assertion_specs.csv \
    --debug_dir results/test_baseline --limit 6

python evaluate_v4.py --debug_dir results/test --designs_csv designs.csv
```

### Full experiment (192 designs / 576 specs, on SLURM)

```bash
sbatch slurm_v4_full192.sh        # 32-task array, 6 designs/task, full pipeline + baseline
sbatch --dependency=afterok:<job_id> slurm_v4_eval.sh
```

### Multi-model comparison / ablation study

```bash
sbatch slurm_v4_multimodel.sh          # pipeline + baseline across 5 models
sbatch slurm_v4_multimodel_eval.sh
sbatch slurm_v4_ablation_36.sh         # no-RAG / no-verify-loop / no-JG-tools / no-tools, 36 designs
```

### Regenerate the paper's tables from a run

```bash
python export_results.py \
    --results_dirs results/full_pipeline results/full_baseline \
    --labels "Pipeline" "Baseline" \
    --designs_csv designs.csv \
    --output_dir results/paper_results/
```

### FVEval Design2SVA benchmark (apples-to-apples with FVEval Table III)

```bash
./submit_fveval.sh                  # 12 SLURM jobs, 5 trials/design, RTL-only (no NL spec)
python rebuild_fveval_csvs.py       # rebuild CSVs + pass@k (set FVEVAL_DATA / FVEVAL_RESULTS
                                     # env vars if your FVEval clone or run dir differ)
```

### AssertionBench (external benchmark)

```bash
cd for_assertion_bench
sbatch slurm_ab.sh          # pipeline
sbatch slurm_ab_eval.sh     # evaluate
```

---

## Results

### 1. Main: pipeline vs. baseline (qwen3.5:35b, 192 designs, 576 specs)

| Method | Syntax | Functionality | Proven | Falsified |
|---|---|---|---|---|
| **Pipeline** | **0.937** | **0.820** | 790 | 260 |
| Baseline | 0.783 | 0.432 | 445 | 246 |

Pipeline wins on 127/192 designs (66.1%), ties 55 (28.6%), loses 10 (5.2%).

### 2. Multi-model comparison

| Model | Pipeline Func | Baseline Func | Improvement |
|---|---|---|---|
| qwen3.5:35b | 0.820 | 0.432 | +89.7% |
| mistral-small3.1:24b | 0.533 | 0.311 | +71.4% |

### 3. Ablation study (qwen3.5:35b, 36 designs, 108 specs)

| Configuration | Syntax | Func | vs. full pipeline |
|---|---|---|---|
| Full pipeline | 0.937 | 0.820 | — |
| No RAG | 0.909 | 0.709 | −14% |
| No verification loop | 0.829 | 0.647 | −21% |
| No JasperGold tools | 0.800 | 0.584 | −29% |
| No tools at all | 0.703 | 0.361 | −56% |

Each component (retrieval, formal-tool grounding, iterative verification) contributes meaningfully;
removing all of them collapses functional correctness by more than half.

### 4. Result by design complexity

| Complexity | Pipeline Func | Baseline Func | Relative gain |
|---|---|---|---|
| Small (1 module) | 0.931 | 0.690 | +34.9% |
| Medium (3–6 modules) | 0.787 | 0.494 | +59.3% |
| Large (11–51 modules) | 0.644 | 0.254 | +153.5% |

The pipeline's advantage over the baseline grows with design complexity.

### 5. FVEval Design2SVA (external benchmark, RTL-only, no NL spec)

Same 192 designs as FVEval's own Design2SVA task (96 pipeline-style + 96 FSM), evaluated on FVEval's
own definition of functional correctness (an assertion that JasperGold can prove, with nothing
falsified) for direct comparison with FVEval's Table III.

Pipeline-style designs (96):

| Method | Syntax@1 | Syntax@5 | Func@1 | Func@5 |
|---|---|---|---|---|
| GPT-4o (FVEval) | 0.862 | 1.000 | 0.104 | 0.427 |
| Gemini-1.5-Pro (FVEval) | 0.665 | 1.000 | 0.175 | 0.500 |
| Gemini-1.5-Flash (FVEval) | 0.969 | 1.000 | 0.125 | 0.498 |
| **Ours (qwen3.5:35b + tools)** | **0.898** | — | **0.412** | **0.896** |

FSM designs (96):

| Method | Syntax@1 | Syntax@5 | Func@1 | Func@5 |
|---|---|---|---|---|
| GPT-4o (FVEval) | 0.993 | 1.000 | 0.373 | 0.900 |
| Gemini-1.5-Pro (FVEval) | 0.556 | 1.000 | 0.427 | 0.906 |
| **Ours (qwen3.5:35b + tools)** | **0.979** | — | **0.860** | **0.958** |

Combined (192): Func@1 = 0.636, Func@5 = 0.927 — roughly 4× GPT-4o on pipeline-style designs and
2.3× on FSM designs, using an open-weight model plus tool augmentation instead of a larger
closed model alone.

### 6. AssertionBench (external benchmark, qwen3.5:35b)

| Method | Designs | Proven | Total assertions | Overall pass rate |
|---|---|---|---|---|
| Pipeline | 86 | 360 | 2,568 | **0.140** |
| Baseline | 121 | 64 | 2,037 | 0.031 |

---

## Citation

```bibtex
@article{mashnoor2026prooofloop,
  title   = {From Language to Logic: Bridging LLMs \& Formal Representations for RTL Assertion Generation},
  author  = {Mashnoor, Nowfel and Kamali, Hadi and Azar, Kimia},
  journal = {arXiv preprint arXiv:2604.23100},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.23100}
}
```

## License

[MIT](LICENSE)
