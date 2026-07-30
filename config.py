"""
RKHS Configuration — all hyperparameters from the paper (Algorithm 1).

Implementation: GPT-4, |K|=50, |G_train|=200, |G_val|=50,
lambda=0.01, mu=5000, N=3, m=5.
"""

# Dataset sizes
N_TRAIN = 200
N_VAL = 50
N_KERNELS = 50

# RKHS synthesis loop (Algorithm 1)
N_ITERATIONS = 3        # N
TOP_M = 5               # m (top-m retrieval)
BATCH_SIZE = 20          # batch sample per iteration

# Scoring function (Eq. 9)
LAMBDA = 0.01            # runtime penalty weight
MU = 5000                # infeasibility penalty

# DAG generation
DAG_MIN_NODES = 20
DAG_MAX_NODES = 80
OP_TYPES = ["ALU", "MUL", "MEM", "CTRL"]
RESOURCE_LIMITS = {"ALU": 2, "MUL": 1, "MEM": 1, "CTRL": 1}
DEFAULT_DURATION = 1     # d(v) = 1

# LLM
LLM_MODEL = "gpt-4"
LLM_TEMPERATURE = 0.7

# Random seed
SEED = 42
