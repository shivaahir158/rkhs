from core.features import rank_u, rank_d
from core.scheduler import list_schedule

def run_cpop(dag, processors):
    memo_u, memo_d = {}, {}
    scores = {
        v: rank_u(dag, v, processors, memo_u) + rank_d(dag, v, processors, memo_d)
        for v in dag.tasks
    }
    return list_schedule(dag, processors, scores)