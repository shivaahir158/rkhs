# RKHS Scheduler

**Retrieval-Augmented Kernel-Based Heuristic Synthesis for DAG Scheduling**

This repository contains the experimental implementation of **RKHS**, a framework that synthesizes interpretable scheduling heuristics for directed acyclic graph (DAG) task scheduling using structural graph features, large language models (LLMs), and Bayesian optimization.

The system automatically constructs priority functions for list scheduling and evaluates them against classical scheduling algorithms such as **HEFT** and **CPOP** across a variety of DAG families.

The goal is not to replace classical algorithms outright, but to investigate whether **machine-generated heuristics can adapt to different graph structures while remaining transparent and interpretable**.

---

# Project Motivation

Many classical DAG scheduling algorithms rely on fixed heuristics that work well in average cases but may not adapt well to different graph structures.

For example:

* **HEFT** prioritizes tasks using upward rank.
* **CPOP** relies on critical path information.

These rules are designed manually and do not change depending on the structure of the workload.

RKHS explores a different idea:

Instead of designing heuristics manually, we attempt to **synthesize priority functions automatically** using:

1. Structural features extracted from DAGs
2. LLM-guided feature selection
3. Bayesian optimization to tune feature weights

The resulting priority functions remain simple and interpretable.

Example priority function:

```
priority = θ1 * rank_u + θ2 * depth + θ3 * fork + θ4 * comm_weight
```

The framework searches for good combinations of features and weights that lead to better scheduling decisions.

---

# Framework Overview

The RKHS pipeline consists of the following steps.

### 1. DAG Generation

Synthetic DAGs are generated to simulate different workload structures.

Supported graph families:

* Random DAGs
* Erdős–Rényi graphs
* Barabási–Albert scale-free graphs
* Watts–Strogatz small-world graphs

Each graph is converted into a scheduling DAG where:

* nodes represent tasks
* edges represent dependencies
* each task has heterogeneous processor execution costs

---

### 2. Feature Extraction

Structural features are extracted from the DAG.

These features capture scheduling-relevant properties of tasks.

Examples include:

* `rank_u` – upward rank used in HEFT
* `rank_d` – downward rank
* `depth` – distance from entry node
* `in_degree` – number of predecessors
* `out_degree` – number of successors
* `fork` – branching factor
* `comm_weight` – communication pressure

These features form the basis for constructing priority functions.

---

### 3. LLM-Guided Feature Selection

An LLM is used to propose combinations of features that may produce effective scheduling heuristics.

The model receives:

* graph size information
* processor count
* graph family characteristics
* feature definitions

It returns a list of candidate features.

Example output:

```
rank_u, depth, fork, comm_weight
```

These features are then used to build a priority template.

---

### 4. Bayesian Optimization

The priority template contains unknown weights:

```
priority = θ1 f1 + θ2 f2 + ... + θk fk
```

Bayesian optimization searches for weight values that minimize average makespan on training DAGs.

The optimized weights are then evaluated on unseen DAGs.

---

### 5. List Scheduling

Using the synthesized priority function, tasks are scheduled using a standard list scheduling procedure.

The resulting schedules are compared against baseline algorithms.

---

# Baselines

The framework compares RKHS against classical scheduling algorithms.

### HEFT

Heterogeneous Earliest Finish Time.

Uses upward rank to determine task priority.

HEFT is widely considered a strong baseline for heterogeneous DAG scheduling.

---

### CPOP

Critical Path on a Processor.

Prioritizes tasks based on the combined upward and downward ranks.

---

# Ablation Studies

To understand the contribution of each component, several ablations are performed.

## Full RKHS

All components enabled.

* LLM feature selection
* Structural features
* Bayesian optimization

This is the complete framework.

---

## RKHS without Bayesian Optimization

LLM feature selection is used, but feature weights are fixed.

This tests whether Bayesian optimization improves heuristic quality.

---

## Fixed Features with Bayesian Optimization

LLM feature selection is disabled.

A manually chosen feature set is used instead.

Bayesian optimization still tunes the weights.

This measures the impact of LLM-generated feature sets.

---

## Fixed Features without Optimization

Both LLM feature selection and Bayesian optimization are removed.

A static handcrafted priority rule is used.

This represents a traditional manually designed heuristic.

---

# Experimental Setup

Experiments are conducted across multiple configurations.

### Task sizes

```
20
50
100
```

### Processor counts

```
2
4
8
```

### DAG families

```
random
erdos_renyi
barabasi_albert
watts_strogatz
```

### Graphs per configuration

```
50
```

Each configuration is split into:

* training DAGs (80%) for optimization
* testing DAGs (20%) for evaluation

---

# Repository Structure

```
rkhs_scheduler/
│
├── baselines/
│   ├── heft.py
│   └── cpop.py
│
├── core/
│   ├── dag.py
│   ├── generator.py
│   ├── features.py
│   └── priority.py
│
├── llm/
│   ├── prompt_builder.py
│   ├── parser.py
│   └── openai_client.py
│
├── optimizer/
│   └── bayes_opt.py
│
├── experiments/
│   ├── run_single.py
│   ├── run_batch.py
│   └── output_manager.py
│
├── results/
│
├── main.py
└── requirements.txt
```

---

# Running Experiments

Install dependencies:

```
pip install -r requirements.txt
```

Run the full experiment:

```
python experiments/run_batch.py
```

The script will automatically:

1. generate DAG datasets
2. run all ablation configurations
3. evaluate baselines
4. store experiment outputs

---

# Output Files

Each run creates a directory inside `results/`.

Example:

```
results/run_YYYYMMDD_HHMMSS/
```

Generated files include:

### config.json

Experiment configuration.

---

### per_dag_results.csv

Per-graph results including:

* graph family
* ablation type
* HEFT makespan
* CPOP makespan
* RKHS makespan

---

### summary.json

Aggregated statistics.

---

### summary.txt

Human-readable summary of the experiment.

Example output:

```
Average HEFT makespan: 394.417
Average CPOP makespan: 439.639
Average RKHS makespan: 409.044
RKHS better than HEFT on: 285 DAGs
RKHS equal to HEFT on: 500 DAGs
```

---

# Notes on Interpretation

RKHS does not necessarily outperform HEFT on every graph.

Instead, the framework demonstrates that:

* automatically synthesized heuristics can match classical algorithms in many cases
* structural feature combinations influence scheduling behavior
* adaptive heuristics may outperform traditional ones on specific graph structures

The purpose of this framework is to explore **automated heuristic synthesis** rather than replace established algorithms.

---

# Research Context

This implementation was developed as part of research on:

**AI-assisted heuristic synthesis for scheduling and design automation.**

The framework explores how machine learning methods can assist in designing algorithms while preserving interpretability and algorithmic transparency.

