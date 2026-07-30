"""
Resource-constrained list scheduling (Section 1.2, Eqs. 1-4).

Iteratively selects ready operations subject to precedence and resource
constraints, using a priority function pi(v) to rank candidates.
"""

import time
from core.dag import DAG


def compute_ready_set(dag, scheduled, cycle):
    """Ready set R(c) at cycle c (Eq. 4)."""
    ready = []
    for v in dag.operations:
        if v in scheduled:
            continue
        preds = dag.predecessors.get(v, [])
        if all(u in scheduled and scheduled[u] + dag.operations[u].duration <= cycle
               for u in preds):
            ready.append(v)
    return ready


def list_schedule(dag, priority_fn, node_features=None):
    """
    Resource-constrained list scheduling (Section 1.2).

    Args:
        dag: The DAG to schedule.
        priority_fn: Function pi(v, features, dag) -> float.
        node_features: Pre-computed node features dict.

    Returns:
        (schedule, makespan, is_feasible)
    """
    schedule = {}
    max_cycles = dag.num_nodes * 10
    cycle = 0

    while len(schedule) < dag.num_nodes and cycle < max_cycles:
        ready = compute_ready_set(dag, schedule, cycle)
        if not ready:
            cycle += 1
            continue

        # Sort by priority descending, ID for tie-breaking
        ready.sort(key=lambda v: (-priority_fn(v, node_features, dag), v))

        # Count active ops per type at this cycle
        active_counts = {}
        for v, start in schedule.items():
            if start <= cycle < start + dag.operations[v].duration:
                t = dag.operations[v].op_type
                active_counts[t] = active_counts.get(t, 0) + 1

        for v in ready:
            if v in schedule:
                continue
            t = dag.operations[v].op_type
            limit = dag.resource_limits.get(t, 1)
            current = active_counts.get(t, 0)
            if current < limit:
                schedule[v] = cycle
                active_counts[t] = current + 1

        cycle += 1

    if schedule:
        makespan = max(schedule[v] + dag.operations[v].duration for v in schedule)
    else:
        makespan = 0

    is_feasible = len(schedule) == dag.num_nodes
    return schedule, makespan, is_feasible


def run_with_timing(dag, priority_fn, node_features=None):
    """Run list scheduling and return (schedule, makespan, runtime_ms, is_feasible)."""
    start = time.perf_counter()
    schedule, makespan, is_feasible = list_schedule(dag, priority_fn, node_features)
    runtime_ms = (time.perf_counter() - start) * 1000
    return schedule, makespan, runtime_ms, is_feasible


def compute_score(makespan, runtime_ms, is_feasible, lam=0.01, mu=5000):
    """J(H; G) = -L(s_H(G)) - lambda * T_run(H,G) - mu * [infeasible]  (Eq. 9)"""
    penalty = 0 if is_feasible else mu
    return -makespan - lam * runtime_ms - penalty
