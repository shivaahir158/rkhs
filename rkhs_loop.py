"""
RKHS Synthesis Loop — Algorithm 1 from the paper.

RAG-Enhanced Kernel-Based Heuristic Synthesis.
Implementation: GPT-4, |K|=50, |G_train|=200, |G_val|=50,
lambda=0.01, mu=5000, N=3, m=5.
"""

import random
import numpy as np
from config import (
    N_ITERATIONS, TOP_M, BATCH_SIZE, LAMBDA, MU,
    LLM_MODEL, LLM_TEMPERATURE,
)
from core.features import extract_node_features
from core.scheduler import run_with_timing, compute_score
from core.retrieval import retrieve_kernels
from core.priority import level_priority, make_weighted_priority
from llm.prompt_builder import build_synthesis_prompt
from llm.parser import parse_llm_response


def evaluate_heuristic(priority_fn, val_dags, val_features):
    """
    Evaluate heuristic H on validation set G_val (Eq. 10).
    J_bar(H) = (1/|G_val|) * sum J(H; G)
    Returns mean score, per-graph results.
    """
    scores = []
    results = []
    for i, dag in enumerate(val_dags):
        features = val_features[i]
        schedule, makespan, runtime_ms, is_feasible = run_with_timing(
            dag, priority_fn, features
        )
        j = compute_score(makespan, runtime_ms, is_feasible, LAMBDA, MU)
        scores.append(j)
        results.append({
            "dag_idx": i,
            "makespan": makespan,
            "runtime_ms": runtime_ms,
            "is_feasible": is_feasible,
            "score": j,
        })

    mean_score = np.mean(scores) if scores else 0
    return mean_score, results


def generate_feedback(current_results, prev_results, iteration):
    """
    Generate feedback F_i from failures/regressions (Step 12).
    Inspired by Self-Refine [6] and ReAct [5].
    """
    feedback_lines = []

    # Identify failure cases
    failures = [r for r in current_results if not r["is_feasible"]]
    if failures:
        feedback_lines.append(
            f"WARNING: {len(failures)} graphs produced infeasible schedules. "
            "Ensure the priority function handles all graph structures."
        )

    # Identify regressions (if we have previous results)
    if prev_results:
        regressions = 0
        for curr, prev in zip(current_results, prev_results):
            if curr["makespan"] > prev["makespan"] * 1.1:
                regressions += 1
        if regressions > 0:
            feedback_lines.append(
                f"REGRESSION: {regressions} graphs have >10% worse makespan "
                "than the previous iteration."
            )

    # Statistics
    makespans = [r["makespan"] for r in current_results]
    feedback_lines.append(
        f"Current iteration {iteration}: "
        f"avg_makespan={np.mean(makespans):.2f}, "
        f"std={np.std(makespans):.2f}, "
        f"min={np.min(makespans):.2f}, max={np.max(makespans):.2f}"
    )

    # Guidance
    high_latency = [r for r in current_results if r["makespan"] > np.mean(makespans) + np.std(makespans)]
    if high_latency:
        feedback_lines.append(
            f"{len(high_latency)} graphs have abnormally high latency. "
            "Consider boosting criticality weight or adding reconvergence awareness."
        )

    return "\n".join(feedback_lines)


def rkhs_synthesis(
    train_dags,
    val_dags,
    kernel_library,
    emb_mean,
    emb_std,
    n_iterations=N_ITERATIONS,
    top_m=TOP_M,
    batch_size=BATCH_SIZE,
    verbose=True,
):
    """
    Algorithm 1: RKHS Synthesis Loop.

    Args:
        train_dags: list of training DAGs (|G_train| = 200)
        val_dags: list of validation DAGs (|G_val| = 50)
        kernel_library: list of Kernel objects (|K| = 50)
        emb_mean, emb_std: normalization stats from kernel construction
        n_iterations: N = 3
        top_m: m = 5
        batch_size: batch sample size per iteration

    Returns:
        H_star: best heuristic (priority function)
        history: list of (H_i, J_i, code, failure_cases)
    """
    # Pre-compute validation features
    val_features = [extract_node_features(dag) for dag in val_dags]

    # Step 1: Initialize history H <- empty
    history = []
    best_heuristic = None
    best_score = float("-inf")
    prev_results = None

    if verbose:
        print("=" * 70)
        print("RKHS Synthesis Loop")
        print(f"N={n_iterations}, m={top_m}, batch={batch_size}")
        print(f"|K|={len(kernel_library)}, |G_train|={len(train_dags)}, |G_val|={len(val_dags)}")
        print("=" * 70)

    # Step 2: for i <- 1 to N do
    for iteration in range(n_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

        # Step 3: Sample batch B ⊆ G_train
        batch = random.sample(train_dags, min(batch_size, len(train_dags)))

        # Steps 4-6: Retrieve kernels for each graph in batch
        all_retrieved = []
        for dag in batch:
            retrieved = retrieve_kernels(dag, kernel_library, emb_mean, emb_std, top_m)
            all_retrieved.extend(retrieved)

        # Step 7: Aggregate kernels K_B = union of K_G
        seen_ids = set()
        aggregated_kernels = []
        for kernel, sim in all_retrieved:
            if kernel.kernel_id not in seen_ids:
                seen_ids.add(kernel.kernel_id)
                aggregated_kernels.append((kernel, sim))

        # Use a representative graph from batch for prompt
        rep_dag = batch[0]
        rep_features = extract_node_features(rep_dag)

        # Step 8: Construct prompt with retrieved kernels
        feedback = None
        if prev_results is not None:
            feedback = generate_feedback(
                history[-1][3] if history else [],
                prev_results,
                iteration,
            )

        prompt = build_synthesis_prompt(
            dag=rep_dag,
            node_features=rep_features,
            retrieved_kernels=aggregated_kernels[:top_m],
            feedback=feedback,
            iteration=iteration,
        )

        # Step 9: Query LLM to produce heuristic code
        if verbose:
            print("  Querying LLM for heuristic synthesis...")

        try:
            from llm.openai_client import query_llm
            llm_response = query_llm(prompt, model=LLM_MODEL, temperature=LLM_TEMPERATURE)
            priority_fn, code_str, success = parse_llm_response(llm_response)
        except Exception as e:
            if verbose:
                print(f"  LLM query failed: {e}")
            success = False

        if not success:
            if verbose:
                print("  LLM synthesis failed, using fallback heuristic")
            # Fallback: use the best kernel template or default
            if aggregated_kernels:
                best_kernel = aggregated_kernels[0][0]
                priority_fn = best_kernel.template_fn
                code_str = best_kernel.template_desc
            else:
                priority_fn = make_weighted_priority(1.0, 0.5, 0.3)
                code_str = "pi(v) = 1.0*crit(v) + 0.5*fanout(v) - 0.3*level(v)"
        else:
            if verbose:
                print("  LLM synthesis successful")

        # Step 10: Evaluate H_i on G_val (Eq. 10)
        mean_score, results = evaluate_heuristic(priority_fn, val_dags, val_features)

        if verbose:
            makespans = [r["makespan"] for r in results]
            runtimes = [r["runtime_ms"] for r in results]
            n_feasible = sum(1 for r in results if r["is_feasible"])
            print(f"  Avg latency: {np.mean(makespans):.2f} (±{np.std(makespans):.2f})")
            print(f"  Avg runtime: {np.mean(runtimes):.2f} ms")
            print(f"  Feasible: {n_feasible}/{len(results)}")
            print(f"  J_bar: {mean_score:.2f}")

        # Step 11: Append to history
        history.append((priority_fn, mean_score, code_str, results))

        # Track best
        if mean_score > best_score:
            best_score = mean_score
            best_heuristic = priority_fn
            if verbose:
                print(f"  ** New best heuristic (J_bar={best_score:.2f})")

        prev_results = results

    # Step 14: Return H* = argmax J_bar
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Best heuristic J_bar = {best_score:.2f}")
        print("=" * 70)

    return best_heuristic, history
