"""
DAG generator for resource-constrained scheduling experiments.

Generates random DAGs with operation types matching the paper's setup:
|G_train|=200, |G_val|=50, nodes in [20,80], op types {ALU, MUL, MEM, CTRL}.
"""

import random
from core.dag import DAG, Operation
from config import OP_TYPES, RESOURCE_LIMITS, DEFAULT_DURATION


def generate_random_dag(num_nodes=40, edge_prob=0.15, seed=None):
    """Generate a random DAG with operation types and resource constraints."""
    rng = random.Random(seed)
    dag = DAG(resource_limits=dict(RESOURCE_LIMITS))

    for i in range(num_nodes):
        op = Operation(id=i, op_type=rng.choice(OP_TYPES), duration=DEFAULT_DURATION)
        dag.add_operation(op)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                dag.add_edge(i, j)

    # Ensure every non-source node has at least one predecessor
    for v in range(1, num_nodes):
        if not dag.predecessors[v]:
            u = rng.randint(0, v - 1)
            dag.add_edge(u, v)

    return dag


def generate_layered_dag(num_layers=8, nodes_per_layer=(3, 8),
                         inter_layer_prob=0.4, skip_prob=0.05, seed=None):
    """Generate a layered DAG (more realistic for HLS dataflow)."""
    rng = random.Random(seed)
    dag = DAG(resource_limits=dict(RESOURCE_LIMITS))
    node_id = 0
    layers = []

    for _ in range(num_layers):
        layer_size = rng.randint(nodes_per_layer[0], nodes_per_layer[1])
        layer = []
        for _ in range(layer_size):
            op = Operation(id=node_id, op_type=rng.choice(OP_TYPES), duration=DEFAULT_DURATION)
            dag.add_operation(op)
            layer.append(node_id)
            node_id += 1
        layers.append(layer)

    for i in range(len(layers) - 1):
        for u in layers[i]:
            for v in layers[i + 1]:
                if rng.random() < inter_layer_prob:
                    dag.add_edge(u, v)
            if i + 2 < len(layers):
                for v in layers[i + 2]:
                    if rng.random() < skip_prob:
                        dag.add_edge(u, v)

    # Ensure all nodes in next layer have at least one predecessor
    for i in range(1, len(layers)):
        for v in layers[i]:
            if not dag.predecessors[v]:
                u = rng.choice(layers[i - 1])
                dag.add_edge(u, v)

    return dag


def generate_fork_join_dag(num_stages=5, width=(2, 6), seed=None):
    """Generate fork-join structured DAG (common in HLS)."""
    rng = random.Random(seed)
    dag = DAG(resource_limits=dict(RESOURCE_LIMITS))
    node_id = 0

    entry = Operation(id=node_id, op_type=rng.choice(OP_TYPES), duration=DEFAULT_DURATION)
    dag.add_operation(entry)
    prev_join = node_id
    node_id += 1

    for _ in range(num_stages):
        w = rng.randint(width[0], width[1])
        parallel_nodes = []
        for _ in range(w):
            op = Operation(id=node_id, op_type=rng.choice(OP_TYPES), duration=DEFAULT_DURATION)
            dag.add_operation(op)
            dag.add_edge(prev_join, node_id)
            parallel_nodes.append(node_id)
            node_id += 1

        join_op = Operation(id=node_id, op_type=rng.choice(OP_TYPES), duration=DEFAULT_DURATION)
        dag.add_operation(join_op)
        for p in parallel_nodes:
            dag.add_edge(p, node_id)
        prev_join = node_id
        node_id += 1

    return dag


def generate_dataset(n_graphs, seed_base=0):
    """Generate a mixed dataset of DAGs with varying structures."""
    rng = random.Random(seed_base)
    dags = []
    generators = [
        ("random", generate_random_dag),
        ("layered", generate_layered_dag),
        ("fork_join", generate_fork_join_dag),
    ]

    for i in range(n_graphs):
        gen_name, gen_fn = generators[i % len(generators)]
        num_nodes = rng.randint(20, 80)
        seed = seed_base + i

        if gen_name == "random":
            dag = gen_fn(num_nodes=num_nodes, seed=seed)
        elif gen_name == "layered":
            num_layers = max(3, num_nodes // 8)
            dag = gen_fn(num_layers=num_layers, seed=seed)
        else:
            num_stages = max(2, num_nodes // 10)
            dag = gen_fn(num_stages=num_stages, seed=seed)

        dags.append(dag)

    return dags
