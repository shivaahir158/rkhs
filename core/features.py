def avg_comp_cost(task, processors):
    return sum(task.comp_costs[p] for p in processors) / len(processors)

def rank_u(dag, v, processors, memo=None):
    if memo is None:
        memo = {}
    if v in memo:
        return memo[v]

    succs = dag.successors.get(v, [])
    base = avg_comp_cost(dag.tasks[v], processors)

    if not succs:
        memo[v] = base
        return base

    memo[v] = base + max(dag.edges[(v, s)] + rank_u(dag, s, processors, memo) for s in succs)
    return memo[v]

def rank_d(dag, v, processors, memo=None):
    if memo is None:
        memo = {}
    if v in memo:
        return memo[v]

    preds = dag.predecessors.get(v, [])
    if not preds:
        memo[v] = 0.0
        return 0.0

    memo[v] = max(
        rank_d(dag, p, processors, memo) +
        avg_comp_cost(dag.tasks[p], processors) +
        dag.edges[(p, v)]
        for p in preds
    )
    return memo[v]

def depth(dag, v, memo=None):
    if memo is None:
        memo = {}
    if v in memo:
        return memo[v]
    preds = dag.predecessors.get(v, [])
    if not preds:
        memo[v] = 0
        return 0
    memo[v] = 1 + max(depth(dag, p, memo) for p in preds)
    return memo[v]

def in_degree(dag, v):
    return len(dag.predecessors.get(v, []))

def out_degree(dag, v):
    return len(dag.successors.get(v, []))

def fork_node(dag, v):
    return 1 if out_degree(dag, v) > 1 else 0

def join_node(dag, v):
    return 1 if in_degree(dag, v) > 1 else 0

def chain_node(dag, v):
    return 1 if in_degree(dag, v) == 1 and out_degree(dag, v) == 1 else 0

def comm_weight(dag, v):
    incoming = sum(dag.edges[(u, v)] for u in dag.predecessors.get(v, []))
    outgoing = sum(dag.edges[(v, w)] for w in dag.successors.get(v, []))
    return incoming + outgoing

def extract_features(dag, processors):
    ru, rd, dp = {}, {}, {}
    features = {}

    for v in dag.tasks:
        features[v] = {
            "rank_u": rank_u(dag, v, processors, ru),
            "rank_d": rank_d(dag, v, processors, rd),
            "depth": depth(dag, v, dp),
            "in_degree": in_degree(dag, v),
            "out_degree": out_degree(dag, v),
            "fork": fork_node(dag, v),
            "join": join_node(dag, v),
            "chain": chain_node(dag, v),
            "comm_weight": comm_weight(dag, v),
        }
    return features