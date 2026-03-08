def build_prompt(num_tasks, num_processors, edge_prob, num_train_graphs, graph_family=None):
    family_guidance = ""
    if graph_family == "barabasi_albert":
        family_guidance = """
Topology hint:
- This is a scale-free graph family.
- Prefer hub-sensitive and branching-aware features such as fork, in_degree, out_degree, or communication-related structure.
"""
    elif graph_family == "watts_strogatz":
        family_guidance = """
Topology hint:
- This is a small-world graph family.
- Prefer clustered/local structural features such as depth, in_degree, and communication-aware structure.
"""
    elif graph_family == "erdos_renyi":
        family_guidance = """
Topology hint:
- This is a random graph family.
- Prefer balanced combinations of rank-based and structural features.
"""
    else:
        family_guidance = """
Topology hint:
- Use a balanced combination of rank-based and structural features.
"""

    return f"""
You are designing an interpretable priority heuristic for DAG scheduling.

Problem setting:
- Number of tasks: {num_tasks}
- Number of processors: {num_processors}
- Edge probability / density proxy: {edge_prob}
- Number of training DAGs: {num_train_graphs}
- Graph family: {graph_family}

{family_guidance}

Goal:
Generate a compact priority function for list scheduling that is:
1. deterministic
2. interpretable
3. different from plain HEFT/CPOP-style upward-rank-only priority
4. suitable for heterogeneous DAG scheduling

Available features:
- rank_u
- rank_d
- depth
- in_degree
- out_degree
- fork
- comm_weight

Hard constraints:
- Do NOT return only rank_u
- Do NOT return only rank_u and rank_d
- Do NOT return a heuristic equivalent to HEFT or CPOP
- Use 3 to 5 features
- Include at least one structural feature from: depth, in_degree, out_degree, fork, comm_weight
- Prefer feature combinations that behave differently across graph topologies

Output rule:
Return ONLY a comma-separated list of feature names.
No prose. No explanation. No equation.

Example valid output:
rank_u,depth,fork,comm_weight
"""