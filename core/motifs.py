"""
Motif extraction and kernel construction (Appendix A.2, A.3, A.4).

Motifs are extracted deterministically from training DAGs by mining
recurring labeled subgraphs: k-hop neighborhoods, high-centrality
subgraphs, reconvergent fanout regions, deep linear chains.
"""

import numpy as np
from collections import Counter
from core.dag import DAG
from core.features import (
    compute_crit, compute_fanout, compute_levels,
    compute_reconvergence, compute_graph_embedding,
)


def extract_motif_stats(dag):
    """
    Extract structural statistics for motif identification.
    Returns a dict of motif-level statistics for the DAG.
    """
    crit = compute_crit(dag)
    fanout = compute_fanout(dag)
    levels = compute_levels(dag)
    reconv = compute_reconvergence(dag)
    n = dag.num_nodes

    crit_vals = list(crit.values())
    fanout_vals = list(fanout.values())
    reconv_vals = list(reconv.values())

    stats = {
        "avg_crit": np.mean(crit_vals),
        "max_crit": np.max(crit_vals),
        "std_crit": np.std(crit_vals),
        "avg_fanout": np.mean(fanout_vals),
        "max_fanout": np.max(fanout_vals),
        "high_fanout_ratio": sum(1 for f in fanout_vals if f >= 3) / n,
        "avg_reconvergence": np.mean(reconv_vals),
        "max_reconvergence": np.max(reconv_vals),
        "reconv_ratio": sum(1 for r in reconv_vals if r > 0) / n,
        "num_levels": max(levels.values()) + 1 if levels else 1,
        "chain_ratio": sum(
            1 for v in dag.operations
            if len(dag.predecessors.get(v, [])) == 1
            and len(dag.successors.get(v, [])) == 1
        ) / n,
        "n_nodes": n,
        "n_edges": len(dag.edges),
        "edge_density": len(dag.edges) / (n * (n - 1) / 2) if n > 1 else 0,
    }
    return stats


def classify_motif_type(stats):
    """
    Classify the dominant motif type of a DAG based on its statistics.
    Returns one of: 'reconvergent', 'fanout_heavy', 'deep_chain',
                    'parallel_heavy', 'mixed'
    """
    if stats["reconv_ratio"] > 0.15 and stats["avg_reconvergence"] > 0.5:
        return "reconvergent"
    elif stats["high_fanout_ratio"] > 0.2 or stats["max_fanout"] >= 5:
        return "fanout_heavy"
    elif stats["chain_ratio"] > 0.4:
        return "deep_chain"
    elif stats["avg_fanout"] > 1.5 and stats["std_crit"] < stats["avg_crit"] * 0.3:
        return "parallel_heavy"
    else:
        return "mixed"


class Kernel:
    """
    A kernel captures (Appendix A.4):
    (i)  structural signature summarizing recurring motifs
    (ii) heuristic template encoding scheduling preferences
    """

    def __init__(self, kernel_id, motif_type, signature, template_fn, template_desc):
        self.kernel_id = kernel_id
        self.motif_type = motif_type
        self.signature = signature          # dict of structural stats
        self.embedding = None               # will be set from signature
        self.template_fn = template_fn      # callable(v, features, dag) -> priority
        self.template_desc = template_desc  # human-readable description

    def set_embedding(self, emb):
        self.embedding = emb

    def __repr__(self):
        return f"Kernel({self.kernel_id}, {self.motif_type})"


# ---------------------------------------------------------------------------
# Heuristic templates for each kernel type (Appendix A.4)
# ---------------------------------------------------------------------------

def reconvergent_template(v, features, dag):
    """Kernel A: Reconvergent Region (Appendix A.4)."""
    f = features[v]
    return 1.0 * f["crit"] + 0.5 * f["reconvergence"] + 0.3 * f["fanout"]


def deep_chain_template(v, features, dag):
    """Kernel B: Deep Critical Chain (Appendix A.4)."""
    f = features[v]
    return 1.0 * f["crit"] - 0.5 * f["slack"]


def fanout_template(v, features, dag):
    """Fanout-heavy: prioritize high-fanout nodes."""
    f = features[v]
    return 0.5 * f["crit"] + 1.0 * f["fanout"] - 0.3 * f["level"]


def parallel_template(v, features, dag):
    """Parallel-heavy: balance criticality and resource pressure."""
    f = features[v]
    return 1.0 * f["crit"] + 0.3 * f["fanout"] - 0.2 * f["level"]


def mixed_template(v, features, dag):
    """Mixed: balanced priority (Eq. 11)."""
    f = features[v]
    return 1.0 * f["crit"] + 0.5 * f["fanout"] - 0.3 * f["level"]


TEMPLATE_MAP = {
    "reconvergent": (reconvergent_template,
                     "pi(v) = alpha*crit(v) + alpha2*R(v) + alpha3*fanout(v)"),
    "deep_chain": (deep_chain_template,
                   "pi(v) = beta1*crit(v) - beta2*slack(v)"),
    "fanout_heavy": (fanout_template,
                     "pi(v) = 0.5*crit(v) + 1.0*fanout(v) - 0.3*level(v)"),
    "parallel_heavy": (parallel_template,
                       "pi(v) = 1.0*crit(v) + 0.3*fanout(v) - 0.2*level(v)"),
    "mixed": (mixed_template,
              "pi(v) = alpha*crit(v) + beta*fanout(v) - gamma*level(v)"),
}


def build_kernel_library(train_dags, n_kernels=50):
    """
    Build kernel library K from training DAGs (Section 1.3).

    Each kernel is constructed by:
    1. Extracting motif statistics from each training DAG
    2. Classifying motif type
    3. Clustering by structural similarity
    4. Creating kernel with signature + template
    """
    from sklearn.cluster import KMeans

    # Extract stats and embeddings for all training DAGs
    all_stats = []
    all_embeddings = []
    all_types = []

    for dag in train_dags:
        stats = extract_motif_stats(dag)
        emb = compute_graph_embedding(dag)
        motif_type = classify_motif_type(stats)
        all_stats.append(stats)
        all_embeddings.append(emb)
        all_types.append(motif_type)

    embeddings_matrix = np.array(all_embeddings)

    # Z-score normalize (Appendix A.1)
    mean = embeddings_matrix.mean(axis=0)
    std = embeddings_matrix.std(axis=0)
    std[std == 0] = 1.0
    embeddings_normed = (embeddings_matrix - mean) / std

    # Cluster into n_kernels groups
    actual_k = min(n_kernels, len(train_dags))
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_normed)

    kernels = []
    for cluster_id in range(actual_k):
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        if not cluster_indices:
            continue

        # Determine dominant motif type in cluster
        cluster_types = [all_types[i] for i in cluster_indices]
        type_counts = Counter(cluster_types)
        dominant_type = type_counts.most_common(1)[0][0]

        # Average signature
        sig = {}
        for key in all_stats[0].keys():
            sig[key] = np.mean([all_stats[i][key] for i in cluster_indices])

        # Centroid embedding
        centroid = kmeans.cluster_centers_[cluster_id]

        template_fn, template_desc = TEMPLATE_MAP[dominant_type]

        kernel = Kernel(
            kernel_id=cluster_id,
            motif_type=dominant_type,
            signature=sig,
            template_fn=template_fn,
            template_desc=template_desc,
        )
        kernel.set_embedding(centroid)
        kernels.append(kernel)

    return kernels, mean, std
