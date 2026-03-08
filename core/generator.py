import random
import networkx as nx
from core.dag import DAG, Task

def generate_random_dag(num_tasks=20, edge_prob=0.2, num_processors=4, seed=None):
    rng = random.Random(seed)

    g = nx.DiGraph()
    g.add_nodes_from(range(num_tasks))

    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if rng.random() < edge_prob:
                g.add_edge(i, j)

    dag = DAG(tasks={}, edges={})

    for node in g.nodes():
        comp_costs = {p: rng.randint(5, 30) for p in range(num_processors)}
        dag.tasks[node] = Task(id=node, comp_costs=comp_costs)
        dag.successors.setdefault(node, [])
        dag.predecessors.setdefault(node, [])

    for u, v in g.edges():
        dag.add_edge(u, v, rng.randint(1, 10))

    return dag