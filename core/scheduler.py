def earliest_start_time(dag, task_id, proc, schedule, proc_free_time):
    preds = dag.predecessors.get(task_id, [])
    pred_ready = 0.0

    for pred in preds:
        pred_proc, pred_start, pred_finish = schedule[pred]
        transfer = 0.0 if pred_proc == proc else dag.edges[(pred, task_id)]
        pred_ready = max(pred_ready, pred_finish + transfer)

    return max(proc_free_time[proc], pred_ready)

def eft(dag, task_id, proc, schedule, proc_free_time):
    est = earliest_start_time(dag, task_id, proc, schedule, proc_free_time)
    finish = est + dag.tasks[task_id].comp_costs[proc]
    return est, finish

def list_schedule(dag, processors, priority_scores):
    unscheduled = set(dag.tasks.keys())
    ready = {v for v in dag.tasks if len(dag.predecessors.get(v, [])) == 0}
    schedule = {}
    proc_free_time = {p: 0.0 for p in processors}

    while unscheduled:
        selected = max(ready, key=lambda x: priority_scores[x])
        ready.remove(selected)

        best_proc = None
        best_start = None
        best_finish = float("inf")

        for p in processors:
            start, finish = eft(dag, selected, p, schedule, proc_free_time)
            if finish < best_finish:
                best_proc = p
                best_start = start
                best_finish = finish

        schedule[selected] = (best_proc, best_start, best_finish)
        proc_free_time[best_proc] = best_finish
        unscheduled.remove(selected)

        for succ in dag.successors.get(selected, []):
            if succ in unscheduled and all(pred in schedule for pred in dag.predecessors[succ]):
                ready.add(succ)

    makespan = max(item[2] for item in schedule.values()) if schedule else 0.0
    return schedule, makespan