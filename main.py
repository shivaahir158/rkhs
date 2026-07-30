"""
RKHS: RAG-Enhanced Kernel-Based Heuristic Synthesis
Main entry point for the full experiment (Tables 1 & 2 from the paper).

Usage:
    python main.py              # Full experiment (requires OpenAI API key)
    python main.py --offline    # Offline mode (no LLM, uses deterministic heuristics)
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if "--offline" in sys.argv:
        from experiments.run_offline import run_offline_experiment
        run_offline_experiment()
    else:
        from experiments.run_batch import run_full_experiment
        run_full_experiment()


if __name__ == "__main__":
    main()
