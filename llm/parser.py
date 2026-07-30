"""
Parse LLM-generated priority functions (Step 5).

Extracts Python code from LLM response and validates it produces
a callable priority function.
"""

import re


def extract_python_code(response):
    """Extract Python code block from LLM response."""
    # Try to find code in ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try ``` ... ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try to find def priority directly
    pattern = r"(def priority\(.*?\):.*?)(?:\n\n|\Z)"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    return None


def validate_priority_fn(code_str):
    """
    Validate and compile a priority function.
    Returns (fn, True) on success, (None, False) on failure.
    """
    if code_str is None:
        return None, False

    namespace = {}
    try:
        exec(code_str, namespace)
    except Exception:
        return None, False

    if "priority" not in namespace:
        return None, False

    fn = namespace["priority"]

    # Quick smoke test
    try:
        test_features = {
            0: {
                "crit": 5.0, "fanout": 2, "level": 0,
                "slack": 1.0, "reconvergence": 0,
                "in_degree": 0, "out_degree": 2, "op_type": "ALU",
            }
        }
        result = fn(0, test_features, None)
        if not isinstance(result, (int, float)):
            return None, False
    except Exception:
        return None, False

    return fn, True


def parse_llm_response(response):
    """
    Full pipeline: extract code -> validate -> return priority function.
    Returns (fn, code_str, success).
    """
    code_str = extract_python_code(response)
    if code_str is None:
        return None, None, False

    fn, success = validate_priority_fn(code_str)
    return fn, code_str, success
