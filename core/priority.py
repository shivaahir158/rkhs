"""
Priority functions for list scheduling (Section 1.5, Eq. 11).

Baseline: pi_base(v) = level(v)
RKHS final: pi(v) = alpha*crit(v) + beta*fanout(v) - gamma*level(v)
"""


def level_priority(v, features, dag):
    """Baseline priority: pi_base(v) = level(v) with deterministic ID tie-breaking."""
    return features[v]["level"]


def make_weighted_priority(alpha=1.0, beta=0.5, gamma=0.3):
    """
    Create the structure-aware priority function (Eq. 11).
    pi(v) = alpha * crit(v) + beta * fanout(v) - gamma * level(v)
    """
    def priority_fn(v, features, dag):
        f = features[v]
        return alpha * f["crit"] + beta * f["fanout"] - gamma * f["level"]
    return priority_fn


def make_priority_from_code(code_str):
    """
    Create a priority function from LLM-generated Python code.
    The code should define: def priority(v, features, dag) -> float
    """
    namespace = {}
    try:
        exec(code_str, namespace)
        if "priority" in namespace:
            return namespace["priority"], True
    except Exception as e:
        pass
    return None, False
