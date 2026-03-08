import os
import json
import csv
from datetime import datetime
from statistics import mean


def create_run_dir(base_dir="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(run_dir, config):
    path = os.path.join(run_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def save_per_dag_results(run_dir, rows):
    if not rows:
        return

    path = os.path.join(run_dir, "per_dag_results.csv")
    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows):
    total = len(rows)

    heft_vals = [float(r["HEFT"]) for r in rows]
    cpop_vals = [float(r["CPOP"]) for r in rows]
    rkhs_vals = [float(r["RKHS_BO"]) for r in rows]

    rkhs_better_than_heft = sum(float(r["RKHS_BO"]) < float(r["HEFT"]) for r in rows)
    rkhs_better_than_cpop = sum(float(r["RKHS_BO"]) < float(r["CPOP"]) for r in rows)
    rkhs_equal_heft = sum(float(r["RKHS_BO"]) == float(r["HEFT"]) for r in rows)
    rkhs_equal_cpop = sum(float(r["RKHS_BO"]) == float(r["CPOP"]) for r in rows)

    summary = {
        "total_dags": total,
        "avg_HEFT": mean(heft_vals) if heft_vals else None,
        "avg_CPOP": mean(cpop_vals) if cpop_vals else None,
        "avg_RKHS_BO": mean(rkhs_vals) if rkhs_vals else None,
        "rkhs_better_than_heft": rkhs_better_than_heft,
        "rkhs_better_than_cpop": rkhs_better_than_cpop,
        "rkhs_equal_heft": rkhs_equal_heft,
        "rkhs_equal_cpop": rkhs_equal_cpop,
    }
    return summary


def save_summary_json(run_dir, summary):
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def save_summary_txt(run_dir, config, summary):
    path = os.path.join(run_dir, "summary.txt")

    text = []
    text.append("RKHS Scheduler Experiment Summary")
    text.append("=" * 40)
    text.append("")
    text.append("Configuration:")
    for k, v in config.items():
        text.append(f"- {k}: {v}")

    text.append("")
    text.append("Results:")
    text.append(f"- Total DAGs tested: {summary['total_dags']}")
    text.append(f"- Average HEFT makespan: {summary['avg_HEFT']:.3f}")
    text.append(f"- Average CPOP makespan: {summary['avg_CPOP']:.3f}")
    text.append(f"- Average RKHS_BO makespan: {summary['avg_RKHS_BO']:.3f}")
    text.append(f"- RKHS_BO better than HEFT on: {summary['rkhs_better_than_heft']} DAGs")
    text.append(f"- RKHS_BO better than CPOP on: {summary['rkhs_better_than_cpop']} DAGs")
    text.append(f"- RKHS_BO equal to HEFT on: {summary['rkhs_equal_heft']} DAGs")
    text.append(f"- RKHS_BO equal to CPOP on: {summary['rkhs_equal_cpop']} DAGs")

    text.append("")
    text.append("Interpretation:")
    if summary["avg_RKHS_BO"] < summary["avg_HEFT"] and summary["avg_RKHS_BO"] < summary["avg_CPOP"]:
        text.append("- RKHS_BO achieved the best average makespan among the compared methods.")
    elif summary["avg_RKHS_BO"] < summary["avg_HEFT"]:
        text.append("- RKHS_BO outperformed HEFT on average, but not necessarily CPOP.")
    elif summary["avg_RKHS_BO"] < summary["avg_CPOP"]:
        text.append("- RKHS_BO outperformed CPOP on average, but not necessarily HEFT.")
    else:
        text.append("- RKHS_BO did not outperform the baselines on average in this run.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))