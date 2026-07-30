"""
HEFT baseline — not used in the paper's experiments directly,
but kept for reference comparison.
"""

from core.priority import make_weighted_priority


def heft_priority(v, features, dag):
    """HEFT-like: prioritize by remaining critical path."""
    return features[v]["crit"]
