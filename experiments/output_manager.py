"""
Output management for experiment results.
"""

import os
import json
from datetime import datetime


def create_run_dir(base_dir="results", prefix="run"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(run_dir, filename, data):
    path = os.path.join(run_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
