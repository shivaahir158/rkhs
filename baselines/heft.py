from core.features import rank_u
from core.scheduler import list_schedule

def run_heft(dag, processors):
    memo = {}
    scores = {v: rank_u(dag, v, processors, memo) for v in dag.tasks}
    return list_schedule(dag, processors, scores)