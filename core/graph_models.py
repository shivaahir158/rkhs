import random
import networkx as nx
from core.dag import DAG, Task


def _nx_to_custom_dag(G, num_processors=4, seed=None):
    """
    Convert a networkx graph into your custom DAG format.
    """
    rng = random.Random(seed)
    dag = DAG(tasks={}, edges={})

    for node in G.nodes():
        comp_costs = {p: rng.randint(5, 30) for p in range(num_processors)}
        dag.tasks[node] = Task(id=node, comp_costs=comp_costs)
        dag.successors.setdefault(node, [])
        dag.predecessors.setdefault(node, [])

    for u, v in G.edges():
        dag.add_edge(u, v, rng.randint(1, 10))

    return dag


def _undirected_to_ordered_dag(G):
    """
    Convert an undirected graph into a DAG by directing edges
    from smaller node id to larger node id.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(G.nodes())

    for u, v in G.edges():
        if u < v:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)

    return dag


def generate_erdos_renyi_dag(num_tasks=20, p=0.2, num_processors=4, seed=None):
    """
    Erdős–Rényi random DAG
    """
    G = nx.erdos_renyi_graph(num_tasks, p, seed=seed)
    dag_nx = _undirected_to_ordered_dag(G)
    return _nx_to_custom_dag(dag_nx, num_processors=num_processors, seed=seed)


def generate_barabasi_albert_dag(num_tasks=20, m=3, num_processors=4, seed=None):
    """
    Barabási–Albert scale-free DAG
    """
    m = max(1, min(m, num_tasks - 1))
    G = nx.barabasi_albert_graph(num_tasks, m, seed=seed)
    dag_nx = _undirected_to_ordered_dag(G)
    return _nx_to_custom_dag(dag_nx, num_processors=num_processors, seed=seed)


def generate_watts_strogatz_dag(num_tasks=20, k=4, p=0.1, num_processors=4, seed=None):
    """
    Watts–Strogatz small-world DAG
    """
    k = max(2, min(k, num_tasks - 1))
    if k % 2 != 0:
        k += 1
        if k >= num_tasks:
            k -= 2

    G = nx.watts_strogatz_graph(num_tasks, k, p, seed=seed)
    dag_nx = _undirected_to_ordered_dag(G)
    return _nx_to_custom_dag(dag_nx, num_processors=num_processors, seed=seed)