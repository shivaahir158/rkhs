"""
Full experiment runner matching the paper's evaluation.

Table 1: RKHS synthesis loop metrics per iteration.
Table 2: Ablation study (Full RKHS, No Retrieval, No Motif, Random Kernel).
"""

import os
import json
import time
import random
import numpy as np
from datetime import datetime

from config import (
    N_TRAIN, N_VAL, N_KERNELS, N_ITERATIONS, TOP_M,
    BATCH_SIZE, LAMBDA, MU, SEED,
)
from core.generator import generate_dataset
from core.features import extract_node_features
from core.motifs import build_kernel_library
from core.retrieval import retrieve_kernels
from core.scheduler import run_with_timing, compute_score
from core.priority import level_priority, make_weighted_priority
from rkhs_loop import rkhs_synthesis, evaluate_heuristic


def run_baseline(val_dags, val_features):
    """Baseline: pi_base(v) = level(v) with deterministic ID tie-breaking."""
    return evaluate_heuristic(level_priority, val_dags, val_features)


def run_full_rkhs(train_dags, val_dags, kernel_library, emb_mean, emb_std):
    """Full RKHS pipeline (Algorithm 1)."""
    best_heuristic, history = rkhs_synthesis(
        train_dags=train_dags,
        val_dags=val_dags,
        kernel_library=kernel_library,
        emb_mean=emb_mean,
        emb_std=emb_std,
    )
    return best_heuristic, history


def run_no_retrieval(train_dags, val_dags, val_features):
    """
    Ablation: No Retrieval — use LLM without kernel retrieval.
    LLM generates heuristic from scratch without structural motifs.
    """
    from llm.prompt_builder import build_synthesis_prompt
    from llm.openai_client import query_llm
    from llm.parser import parse_llm_response
    from config import LLM_MODEL, LLM_TEMPERATURE

    best_score = float("-inf")
    best_fn = None
    history = []

    for iteration in range(N_ITERATIONS):
        rep_dag = random.choice(train_dags)
        rep_features = extract_node_features(rep_dag)

        prompt = build_synthesis_prompt(
            dag=rep_dag,
            node_features=rep_features,
            retrieved_kernels=[],  # NO retrieval
            iteration=iteration,
        )

        try:
            response = query_llm(prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            fn, code, success = parse_llm_response(response)
        except Exception:
            success = False

        if not success:
            fn = make_weighted_priority(1.0, 0.5, 0.3)
            code = "fallback"

        score, results = evaluate_heuristic(fn, val_dags, val_features)
        history.append((fn, score, code, results))
        if score > best_score:
            best_score = score
            best_fn = fn

    return best_fn, history


def run_no_motif(train_dags, val_dags, val_features, kernel_library, emb_mean, emb_std):
    """
    Ablation: No Motif — retrieve kernels but strip motif signatures.
    Only use structural embedding for retrieval, no template guidance.
    """
    from llm.prompt_builder import build_synthesis_prompt
    from llm.openai_client import query_llm
    from llm.parser import parse_llm_response
    from config import LLM_MODEL, LLM_TEMPERATURE

    best_score = float("-inf")
    best_fn = None
    history = []

    for iteration in range(N_ITERATIONS):
        batch = random.sample(train_dags, min(BATCH_SIZE, len(train_dags)))

        # Retrieve kernels but clear their template descriptions
        all_retrieved = []
        for dag in batch:
            retrieved = retrieve_kernels(dag, kernel_library, emb_mean, emb_std, TOP_M)
            # Strip motif info — only keep similarity score
            stripped = []
            for kernel, sim in retrieved:
                from core.motifs import Kernel, mixed_template
                stripped_k = Kernel(
                    kernel_id=kernel.kernel_id,
                    motif_type="unknown",
                    signature={k: 0.0 for k in kernel.signature},
                    template_fn=mixed_template,
                    template_desc="(motif abstracted away)",
                )
                stripped_k.set_embedding(kernel.embedding)
                stripped.append((stripped_k, sim))
            all_retrieved.extend(stripped)

        rep_dag = batch[0]
        rep_features = extract_node_features(rep_dag)

        prompt = build_synthesis_prompt(
            dag=rep_dag,
            node_features=rep_features,
            retrieved_kernels=all_retrieved[:TOP_M],
            iteration=iteration,
        )

        try:
            response = query_llm(prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            fn, code, success = parse_llm_response(response)
        except Exception:
            success = False

        if not success:
            fn = make_weighted_priority(1.0, 0.5, 0.3)
            code = "fallback"

        score, results = evaluate_heuristic(fn, val_dags, val_features)
        history.append((fn, score, code, results))
        if score > best_score:
            best_score = score
            best_fn = fn

    return best_fn, history


def run_random_kernel(train_dags, val_dags, val_features, kernel_library, emb_mean, emb_std):
    """
    Ablation: Random Kernel — retrieve kernels randomly instead of by similarity.
    """
    from llm.prompt_builder import build_synthesis_prompt
    from llm.openai_client import query_llm
    from llm.parser import parse_llm_response
    from config import LLM_MODEL, LLM_TEMPERATURE

    best_score = float("-inf")
    best_fn = None
    history = []

    for iteration in range(N_ITERATIONS):
        rep_dag = random.choice(train_dags)
        rep_features = extract_node_features(rep_dag)

        # Random kernel selection instead of similarity-based
        random_kernels = random.sample(
            kernel_library, min(TOP_M, len(kernel_library))
        )
        random_retrieved = [(k, 0.0) for k in random_kernels]

        prompt = build_synthesis_prompt(
            dag=rep_dag,
            node_features=rep_features,
            retrieved_kernels=random_retrieved,
            iteration=iteration,
        )

        try:
            response = query_llm(prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            fn, code, success = parse_llm_response(response)
        except Exception:
            success = False

        if not success:
            fn = make_weighted_priority(1.0, 0.5, 0.3)
            code = "fallback"

        score, results = evaluate_heuristic(fn, val_dags, val_features)
        history.append((fn, score, code, results))
        if score > best_score:
            best_score = score
            best_fn = fn

    return best_fn, history


def print_table1(history):
    """
    Print Table 1: RKHS synthesis loop metrics per iteration.
    Columns: Iter, Latency, Runtime (ms), J_bar
    """
    print("\n" + "=" * 65)
    print("Table 1: RKHS synthesis loop — average metrics per iteration")
    print("=" * 65)
    print(f"{'Iter':<25s} {'Latency↓':>10s} {'Runtime(ms)↓':>14s} {'J_bar↑':>10s}")
    print("-" * 65)

    labels = [
        "0 (Topo+ID)",
        "1 (Fanout-aware)",
        "2 (Zero-in PQ)",
    ]

    for i, (fn, score, code, results) in enumerate(history):
        makespans = [r["makespan"] for r in results]
        runtimes = [r["runtime_ms"] for r in results]
        label = labels[i] if i < len(labels) else f"{i}"
        print(f"{label:<25s} {np.mean(makespans):>10.2f} {np.mean(runtimes):>14.2f} {score:>10.2f}")
    print("=" * 65)


def print_table2(ablation_results, val_dags):
    """
    Print Table 2: Ablation study results.
    Columns: Graph, Full RKHS, No Retrieval, No Motif, Random Kernel
    """
    print("\n" + "=" * 75)
    print("Table 2: Ablation study — latency across validation graphs")
    print("=" * 75)
    print(f"{'Graph':<10s} {'Full RKHS':>12s} {'No Retrieval':>14s} {'No Motif':>12s} {'Random Kernel':>15s}")
    print("-" * 75)

    methods = ["full_rkhs", "no_retrieval", "no_motif", "random_kernel"]
    # Show representative graphs (G1-G4)
    n_show = min(4, len(val_dags))
    for g_idx in range(n_show):
        row = f"G{g_idx + 1:<9d}"
        for method in methods:
            results = ablation_results[method]
            if g_idx < len(results):
                row += f" {results[g_idx]['makespan']:>12.1f}"
            else:
                row += f" {'N/A':>12s}"
        print(row)

    # Average row
    row = f"{'Avg':<10s}"
    for method in methods:
        results = ablation_results[method]
        makespans = [r["makespan"] for r in results]
        std = np.std(makespans)
        row += f" {np.mean(makespans):>6.2f}(±{std:.2f})"
    print(row)
    print("=" * 75)


def run_full_experiment():
    """
    Run the complete RKHS experiment matching the paper.
    Produces Tables 1 and 2.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    print("Generating datasets...")
    train_dags = generate_dataset(N_TRAIN, seed_base=0)
    val_dags = generate_dataset(N_VAL, seed_base=10000)

    print(f"Generated {len(train_dags)} training DAGs and {len(val_dags)} validation DAGs")

    # Pre-compute validation features
    print("Extracting features...")
    val_features = [extract_node_features(dag) for dag in val_dags]

    # Build kernel library
    print(f"Building kernel library (|K|={N_KERNELS})...")
    kernel_library, emb_mean, emb_std = build_kernel_library(train_dags, N_KERNELS)
    print(f"Built {len(kernel_library)} kernels")

    # --- Baseline ---
    print("\nRunning baseline (level-based priority)...")
    baseline_score, baseline_results = run_baseline(val_dags, val_features)
    baseline_makespans = [r["makespan"] for r in baseline_results]
    print(f"Baseline avg latency: {np.mean(baseline_makespans):.2f} "
          f"(±{np.std(baseline_makespans):.2f})")

    # --- Full RKHS (Table 1) ---
    print("\nRunning Full RKHS synthesis (Algorithm 1)...")
    best_heuristic, rkhs_history = run_full_rkhs(
        train_dags, val_dags, kernel_library, emb_mean, emb_std
    )
    print_table1(rkhs_history)

    # --- Ablation Study (Table 2) ---
    print("\nRunning ablation studies...")
    ablation_results = {}

    # Full RKHS results (best iteration)
    best_iter_idx = max(range(len(rkhs_history)), key=lambda i: rkhs_history[i][1])
    ablation_results["full_rkhs"] = rkhs_history[best_iter_idx][3]

    # No Retrieval
    print("  Running No Retrieval ablation...")
    _, no_retr_history = run_no_retrieval(train_dags, val_dags, val_features)
    best_idx = max(range(len(no_retr_history)), key=lambda i: no_retr_history[i][1])
    ablation_results["no_retrieval"] = no_retr_history[best_idx][3]

    # No Motif
    print("  Running No Motif ablation...")
    _, no_motif_history = run_no_motif(
        train_dags, val_dags, val_features, kernel_library, emb_mean, emb_std
    )
    best_idx = max(range(len(no_motif_history)), key=lambda i: no_motif_history[i][1])
    ablation_results["no_motif"] = no_motif_history[best_idx][3]

    # Random Kernel
    print("  Running Random Kernel ablation...")
    _, rand_history = run_random_kernel(
        train_dags, val_dags, val_features, kernel_library, emb_mean, emb_std
    )
    best_idx = max(range(len(rand_history)), key=lambda i: rand_history[i][1])
    ablation_results["random_kernel"] = rand_history[best_idx][3]

    print_table2(ablation_results, val_dags)

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config = {
        "N_TRAIN": N_TRAIN, "N_VAL": N_VAL, "N_KERNELS": N_KERNELS,
        "N_ITERATIONS": N_ITERATIONS, "TOP_M": TOP_M, "BATCH_SIZE": BATCH_SIZE,
        "LAMBDA": LAMBDA, "MU": MU, "SEED": SEED,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save iteration results (Table 1)
    table1_data = []
    for i, (fn, score, code, results) in enumerate(rkhs_history):
        makespans = [r["makespan"] for r in results]
        runtimes = [r["runtime_ms"] for r in results]
        table1_data.append({
            "iteration": i,
            "avg_latency": float(np.mean(makespans)),
            "std_latency": float(np.std(makespans)),
            "avg_runtime_ms": float(np.mean(runtimes)),
            "j_bar": float(score),
            "code": str(code),
        })
    with open(os.path.join(run_dir, "table1_iterations.json"), "w") as f:
        json.dump(table1_data, f, indent=2)

    # Save ablation results (Table 2)
    table2_data = {}
    for method, results in ablation_results.items():
        makespans = [r["makespan"] for r in results]
        table2_data[method] = {
            "avg_latency": float(np.mean(makespans)),
            "std_latency": float(np.std(makespans)),
            "per_graph": [{"dag_idx": r["dag_idx"], "makespan": r["makespan"]}
                          for r in results[:4]],
        }
    table2_data["baseline"] = {
        "avg_latency": float(np.mean(baseline_makespans)),
        "std_latency": float(np.std(baseline_makespans)),
    }
    with open(os.path.join(run_dir, "table2_ablation.json"), "w") as f:
        json.dump(table2_data, f, indent=2)

    # Improvement over baseline
    rkhs_avg = np.mean([r["makespan"] for r in ablation_results["full_rkhs"]])
    baseline_avg = np.mean(baseline_makespans)
    improvement = (baseline_avg - rkhs_avg) / baseline_avg * 100

    print(f"\n{'=' * 65}")
    print("Summary")
    print(f"{'=' * 65}")
    print(f"Baseline avg latency: {baseline_avg:.2f}")
    print(f"Full RKHS avg latency: {rkhs_avg:.2f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Results saved to: {run_dir}")

    return run_dir


if __name__ == "__main__":
    run_full_experiment()
