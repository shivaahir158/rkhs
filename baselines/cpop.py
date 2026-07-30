"""
CPOP baseline — not used in the paper's experiments directly,
but kept for reference comparison.
"""


def cpop_priority(v, features, dag):
    """CPOP-like: crit + level (approximation of rank_u + rank_d)."""
    return features[v]["crit"] + features[v]["level"]
