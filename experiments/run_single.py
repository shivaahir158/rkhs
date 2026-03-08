from baselines.heft import run_heft
from baselines.cpop import run_cpop
from core.scheduler import list_schedule


def evaluate_single_dag(dag, processors, features, template, theta, llm_response):
    _, heft_ms = run_heft(dag, processors)
    _, cpop_ms = run_cpop(dag, processors)

    rkhs_scores = {
        v: template.score(features[v], theta)
        for v in dag.tasks
    }
    _, rkhs_ms = list_schedule(dag, processors, rkhs_scores)

    return {
        "HEFT": heft_ms,
        "CPOP": cpop_ms,
        "RKHS_BO": rkhs_ms,
        "theta": theta,
        "llm_response": llm_response,
        "template_features": template.feature_names,
    }