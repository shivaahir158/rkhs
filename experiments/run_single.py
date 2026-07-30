"""
Single DAG evaluation utility.
"""

from core.features import extract_node_features
from core.scheduler import run_with_timing, compute_score
from core.priority import level_priority
from config import LAMBDA, MU


def evaluate_single_dag(dag, priority_fn, node_features=None):
    """Evaluate a single DAG with a given priority function."""
    if node_features is None:
        node_features = extract_node_features(dag)

    schedule, makespan, runtime_ms, is_feasible = run_with_timing(
        dag, priority_fn, node_features
    )
    score = compute_score(makespan, runtime_ms, is_feasible, LAMBDA, MU)

    # Also run baseline for comparison
    _, base_makespan, base_runtime, base_feasible = run_with_timing(
        dag, level_priority, node_features
    )

    return {
        "makespan": makespan,
        "runtime_ms": runtime_ms,
        "is_feasible": is_feasible,
        "score": score,
        "baseline_makespan": base_makespan,
        "improvement": (base_makespan - makespan) / base_makespan * 100
        if base_makespan > 0 else 0,
    }
