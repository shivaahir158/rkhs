# RKHS: RAG-Enhanced Kernel-Based Heuristic Synthesis

**A Structured Methodology Using Large Language Models for Hardware Design**

Implementation of the RKHS framework, check out our work at [arXiv:2604.26153v1](https://arxiv.org/abs/2604.26153), which synthesizes reusable optimization heuristics for resource-constrained list scheduling in high-level synthesis (HLS) using retrieval-augmented generation (RAG), compact kernel heuristic templates, and an LLM-driven refinement loop.

---

> **If you find our work useful and want to build on it for your own ideas, please cite us:**
>
> ```bibtex
> @article{ahir2025rkhs,
>   title={RAG-Enhanced Kernel-Based Heuristic Synthesis (RKHS):
>          A Structured Methodology Using Large Language Models
>          for Hardware Design},
>   author={Ahir, Shiva and Doboli, Alex},
>   journal={arXiv preprint arXiv:2604.26153},
>   year={2026}
> }
> ```
>
> We'd love to hear how you're using RKHS. Suggestions, feedback, and collaboration ideas are always welcome, feel free to reach out at **shiva.ahir@stonybrook.edu**

---

## Key Results

| Metric | Value |
|--------|-------|
| Baseline avg latency | 17.30 |
| RKHS avg latency | 16.02 |
| Improvement | **7.4%** |
| Runtime overhead | **1.3x** |

**Synthesized priority function (Eq. 11):**

```
π(v) = α · crit(v) + β · fanout(v) − γ · level(v)
```

**Best LLM-generated heuristic (GPT-4):**

```python
def priority(v, features, dag):
    f = features[v]
    return 0.7 * f["crit"] + 0.2 * f["reconvergence"] + 0.1 * f["fanout"]
```

---

## Problem Formulation (Section 1.2)

We consider latency-minimizing list scheduling for HLS — a resource-constrained scheduling problem over a DAG `G = (V, E)`:

- Each node `v` is an operation with type `τ(v) ∈ {ALU, MUL, MEM, CTRL}` and duration `d(v) = 1`
- Each edge `(u, v)` encodes precedence `u ≺ v`
- Resource capacities `R_t` limit how many type-t operations execute per cycle
- **Objective:** minimize makespan `L(s) = max(s(v) + d(v))`

**Baseline:** `π_base(v) = level(v)` with deterministic ID tie-breaking.

---

## Framework Overview (Algorithm 1)

```
Input DAG → Feature/Embedding f(G) → Kernel Retrieval top-m → LLM Synthesis → Schedule & Score
```

The RKHS pipeline:

1. **Generate training DAGs** (`|G_train| = 200`) and validation DAGs (`|G_val| = 50`)
2. **Build kernel library** (`|K| = 50`) by extracting motifs from training DAGs and clustering by structural similarity
3. **For each iteration** (`N = 3`):
   - Sample a batch from training DAGs
   - Retrieve top-`m` kernels via cosine similarity (Eq. 5-6)
   - Construct RAG-augmented prompt with kernel motifs and templates
   - Query GPT-4 to synthesize a priority function
   - Evaluate on validation set using scoring function `J(H; G)` (Eq. 9)
   - Generate feedback for next iteration (Self-Refine)
4. **Return** the best heuristic `H*`

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `\|K\|` | 50 | Kernel library size |
| `\|G_train\|` | 200 | Training graphs |
| `\|G_val\|` | 50 | Validation graphs |
| `N` | 3 | Synthesis iterations |
| `m` | 5 | Top-m kernel retrieval |
| `λ` | 0.01 | Runtime penalty weight |
| `μ` | 5000 | Infeasibility penalty |

---

## Kernel Library (Appendix A)

### Deterministic Embedding (A.1)

```
f(G) = [crit-path summary, fanout histogram, level histogram, op-type histogram, resource-pressure proxies]
```

All features are z-score normalized over `G_train`.

### Motif Extraction (A.2)

Motifs are extracted by mining recurring labeled subgraphs:
- k-hop neighborhoods
- High-centrality subgraphs
- Reconvergent fanout regions
- Deep linear chains

### Kernel Templates (A.4)

**Kernel A — Reconvergent Region:**
```
π(v) = α₁ · crit(v) + α₂ · R(v) + α₃ · fanout(v)
```

**Kernel B — Deep Critical Chain:**
```
π(v) = β₁ · crit(v) − β₂ · slack(v)
```

### Reconvergence Marker (A.3)

```
R(v) = |{(u, w) : u, w ∈ children(v), ∃x s.t. u → x ∧ w → x}|
```

---

## Node-Level Features

| Feature | Description |
|---------|-------------|
| `crit(v)` | Remaining critical-path length to sink (Eq. 12) |
| `fanout(v)` | Number of successors |
| `level(v)` | Longest path from source |
| `slack(v)` | ALAP − ASAP scheduling flexibility |
| `R(v)` | Reconvergence marker |
| `in_degree` | Number of predecessors |
| `out_degree` | Number of successors |
| `op_type` | Operation type (ALU, MUL, MEM, CTRL) |

---

## Scoring Function (Eq. 9)

```
J(H; G) = −L(s_H(G)) − λ · T_run(H, G) − μ · [infeasible]
```

The mean validation score:

```
J̄(H) = (1/|G_val|) · Σ J(H; G)
```

---

## Experiment Results

### Table 1: RKHS Synthesis Loop (per iteration)

| Iter | Latency ↓ | Runtime (ms) ↓ | J̄ ↑ |
|------|-----------|----------------|------|
| 0 (Topo+ID) | 16.14 | 0.18 | -16.14 |
| 1 (Fanout-aware) | **16.02** | 0.18 | **-16.02** |
| 2 (Zero-in PQ) | 16.12 | 0.20 | -16.12 |

### Table 2: Ablation Study

| Graph | Full RKHS | No Retrieval | No Motif | Random Kernel |
|-------|-----------|--------------|----------|---------------|
| G1 | 19.0 | 19.0 | 19.0 | 19.0 |
| G2 | 17.0 | 17.0 | 17.0 | 20.0 |
| G3 | 16.0 | 16.0 | 16.0 | 16.0 |
| G4 | 13.0 | 13.0 | 13.0 | 15.0 |
| **Avg** | **15.94 (±6.83)** | 15.96 (±6.81) | 16.04 (±6.90) | 16.38 (±6.99) |

---

## Repository Structure

```
rkhs_scheduler/
├── config.py                  # All hyperparameters (Algorithm 1)
├── main.py                    # Entry point (--offline for no-API mode)
├── rkhs_loop.py               # Algorithm 1: RKHS synthesis loop
│
├── core/
│   ├── dag.py                 # DAG with Operation types, resource limits
│   ├── generator.py           # DAG generation (random, layered, fork-join)
│   ├── features.py            # Node features + graph embedding f(G)
│   ├── motifs.py              # Motif extraction, kernel construction
│   ├── retrieval.py           # Cosine similarity kernel retrieval
│   ├── scheduler.py           # Resource-constrained list scheduling
│   └── priority.py            # Priority functions (baseline + synthesized)
│
├── llm/
│   ├── prompt_builder.py      # RAG-augmented prompt construction
│   ├── parser.py              # Parse/validate LLM-generated code
│   └── openai_client.py       # GPT-4 API client
│
├── baselines/
│   ├── heft.py                # HEFT-style baseline (reference)
│   └── cpop.py                # CPOP-style baseline (reference)
│
├── experiments/
│   ├── run_batch.py           # Full experiment (Tables 1 & 2)
│   ├── run_offline.py         # Offline mode (no API key needed)
│   ├── run_single.py          # Single DAG evaluation
│   └── output_manager.py      # Result file management
│
├── results/                   # Experiment outputs (JSON)
├── requirements.txt
└── .env                       # OpenAI API key (not committed)
```

---

## Running Experiments

### Install dependencies

```bash
pip install -r requirements.txt
```

### Full experiment (requires OpenAI API key)

```bash
# Set your API key in .env
echo "OPENAI_API_KEY=sk-..." > .env

# Run full RKHS with GPT-4 synthesis
python3 main.py
```

### Offline experiment (no API key needed)

```bash
python3 main.py --offline
```

Uses deterministic heuristic templates from the kernel library instead of LLM synthesis.

---

## Output Files

Each run creates a timestamped directory in `results/`:

```
results/run_YYYYMMDD_HHMMSS/
├── config.json                # Experiment hyperparameters
├── table1_iterations.json     # Per-iteration metrics (Table 1)
└── table2_ablation.json       # Ablation study results (Table 2)
```

---

## Ablation Study Design

| Ablation | Retrieval | Motifs | LLM |
|----------|-----------|--------|-----|
| **Full RKHS** | Similarity-based | Full signatures + templates | Yes |
| **No Retrieval** | None | None | Yes (no kernel context) |
| **No Motif** | Similarity-based | Signatures stripped | Yes |
| **Random Kernel** | Random selection | Full signatures + templates | Yes |
```
