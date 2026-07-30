"""
Similarity-based kernel retrieval (Section 1.4, Eqs. 5-6).

Given a new graph G, compute its deterministic embedding f(G),
then retrieve the top-m most similar kernels via cosine similarity.
"""

import numpy as np
from core.features import compute_graph_embedding


def cosine_similarity(a, b):
    """Cosine similarity sim(G, K) = <f(G), f(K)> / (||f(G)|| * ||f(K)||)  (Eq. 5)"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_kernels(dag, kernel_library, emb_mean, emb_std, top_m=5):
    """
    Retrieve top-m kernels for a given DAG (Eq. 6).
    K_G = TopM_{K in K} sim(G, K)
    """
    # Compute embedding and normalize using training stats
    emb = compute_graph_embedding(dag)
    emb_normed = (emb - emb_mean) / emb_std

    similarities = []
    for kernel in kernel_library:
        sim = cosine_similarity(emb_normed, kernel.embedding)
        similarities.append((kernel, sim))

    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_m]
