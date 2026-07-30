"""
Feature extraction for RKHS (Sections 1.2, 1.4, A.1).

Node-level features: crit(v), fanout(v), level(v), slack(v), R(v).
Graph-level embedding: f(G) for retrieval (Appendix A.1).
"""

import numpy as np
from collections import Counter
from core.dag import DAG


def compute_levels(dag):
    """Level of each node (longest path from any source)."""
    levels = {}
    topo = dag.topological_order()
    for v in topo:
        preds = dag.predecessors.get(v, [])
        if not preds:
            levels[v] = 0
        else:
            levels[v] = 1 + max(levels[u] for u in preds)
    return levels


def compute_crit(dag):
    """
    Remaining critical-path length from v to sink (Eq. 12).
    crit(v) = max over paths from v to sink of sum of d(u).
    """
    depths = {}
    topo = dag.topological_order()
    for v in reversed(topo):
        succs = dag.successors.get(v, [])
        d_v = dag.operations[v].duration
        if not succs:
            depths[v] = d_v
        else:
            depths[v] = d_v + max(depths[s] for s in succs)
    return depths


def compute_fanout(dag):
    """fanout(v) = number of successors."""
    return {v: len(dag.successors.get(v, [])) for v in dag.operations}


def compute_slack(dag):
    """slack(v) = ALAP(v) - ASAP(v), without resource constraints."""
    asap = {}
    topo = dag.topological_order()
    for v in topo:
        preds = dag.predecessors.get(v, [])
        if not preds:
            asap[v] = 0
        else:
            asap[v] = max(asap[u] + dag.operations[u].duration for u in preds)

    makespan = max(asap[v] + dag.operations[v].duration for v in dag.operations)

    alap = {}
    for v in reversed(topo):
        succs = dag.successors.get(v, [])
        if not succs:
            alap[v] = makespan - dag.operations[v].duration
        else:
            alap[v] = min(alap[s] for s in succs) - dag.operations[v].duration

    return {v: alap[v] - asap[v] for v in dag.operations}


def compute_reconvergence(dag):
    """
    Reconvergence marker R(v) (Appendix A.3).
    R(v) = |{(u,w) : u,w in children(v), exists x s.t. u->x and w->x}|
    """
    reconv = {}
    for v in dag.operations:
        children = dag.successors.get(v, [])
        count = 0
        if len(children) > 1:
            child_descs = {}
            for c in children:
                visited = set()
                queue = [c]
                for _ in range(5):  # limited depth BFS
                    next_q = []
                    for node in queue:
                        if node not in visited:
                            visited.add(node)
                            next_q.extend(dag.successors.get(node, []))
                    queue = next_q
                    if not queue:
                        break
                child_descs[c] = visited

            for i, u in enumerate(children):
                for w in children[i + 1:]:
                    if child_descs.get(u, set()) & child_descs.get(w, set()):
                        count += 1
        reconv[v] = count
    return reconv


def extract_node_features(dag):
    """Extract all node-level features for each node in the DAG."""
    levels = compute_levels(dag)
    crit = compute_crit(dag)
    fanout = compute_fanout(dag)
    slack = compute_slack(dag)
    reconv = compute_reconvergence(dag)

    features = {}
    for v in dag.operations:
        features[v] = {
            "level": levels[v],
            "crit": crit[v],
            "fanout": fanout[v],
            "slack": slack[v],
            "reconvergence": reconv[v],
            "in_degree": len(dag.predecessors.get(v, [])),
            "out_degree": len(dag.successors.get(v, [])),
            "op_type": dag.operations[v].op_type,
        }
    return features


def compute_graph_embedding(dag):
    """
    Deterministic embedding f(G) in R^d (Appendix A.1).
    f(G) = [crit-path summary, fanout histogram, level histogram,
            op-type histogram, resource-pressure proxies].
    """
    crit = compute_crit(dag)
    fanout = compute_fanout(dag)
    levels = compute_levels(dag)
    n = dag.num_nodes

    # Crit-path summary (5 features)
    crit_vals = list(crit.values())
    crit_summary = [
        np.mean(crit_vals), np.std(crit_vals),
        np.max(crit_vals), np.min(crit_vals),
        np.max(crit_vals) - np.min(crit_vals),
    ]

    # Fanout histogram (5 bins: 0, 1, 2, 3, 4+)
    fanout_hist = [0] * 5
    for f in fanout.values():
        fanout_hist[min(f, 4)] += 1
    fanout_hist = [x / n for x in fanout_hist]

    # Level histogram (5 bins, normalized)
    level_vals = list(levels.values())
    max_level = max(level_vals) if level_vals else 1
    level_hist = [0] * 5
    for lv in level_vals:
        idx = min(int(lv / (max_level + 1) * 5), 4)
        level_hist[idx] += 1
    level_hist = [x / n for x in level_hist]

    # Op-type histogram
    op_types_all = ["ALU", "MUL", "MEM", "CTRL"]
    type_counts = Counter(dag.operations[v].op_type for v in dag.operations)
    op_hist = [type_counts.get(t, 0) / n for t in op_types_all]

    # Resource-pressure proxies
    res_pressure = []
    for t in op_types_all:
        count = type_counts.get(t, 0)
        limit = dag.resource_limits.get(t, 1)
        res_pressure.append(count / (n * limit) if limit > 0 else 0)

    return np.array(
        crit_summary + fanout_hist + level_hist + op_hist + res_pressure,
        dtype=np.float64,
    )
