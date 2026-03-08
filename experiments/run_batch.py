from core.generator import generate_random_dag
from core.graph_models import (
    generate_erdos_renyi_dag,
    generate_barabasi_albert_dag,
    generate_watts_strogatz_dag,
)
from core.features import extract_features
from core.priority import PriorityTemplate

from llm.prompt_builder import build_prompt
from llm.parser import parse_feature_names
from llm.openai_client import ask_llm

from optimizer.bayes_opt import optimize_theta_on_dataset
from experiments.run_single import evaluate_single_dag
from experiments.output_manager import (
    create_run_dir,
    save_config,
    save_per_dag_results,
    build_summary,
    save_summary_json,
    save_summary_txt,
)


def get_ablation_configs():
    return [
        {
            "name": "full_rkhs",
            "use_llm": True,
            "use_bo": True,
            "fixed_feature_names": None,
            "fixed_theta": None,
        },
        {
            "name": "no_bo",
            "use_llm": True,
            "use_bo": False,
            "fixed_feature_names": None,
            "fixed_theta": None,
        },
        {
            "name": "fixed_features_bo",
            "use_llm": False,
            "use_bo": True,
            "fixed_feature_names": ["rank_u", "rank_d", "fork", "comm_weight"],
            "fixed_theta": None,
        },
        {
            "name": "fixed_features_no_bo",
            "use_llm": False,
            "use_bo": False,
            "fixed_feature_names": ["rank_u", "rank_d", "fork", "comm_weight"],
            "fixed_theta": None,
        },
    ]


def get_graph_family_configs(edge_prob):
    return [
        {
            "name": "random",
            "params": {"edge_prob": edge_prob},
        },
        {
            "name": "erdos_renyi",
            "params": {"p": edge_prob},
        },
        {
            "name": "barabasi_albert",
            "params": {"m": 3},
        },
        {
            "name": "watts_strogatz",
            "params": {"k": 4, "p": 0.1},
        },
    ]


def default_theta_for_template(template):
    return {feature: 1.0 for feature in template.feature_names}


def build_dag(graph_family, graph_params, num_tasks, num_processors, edge_prob, seed):
    if graph_family == "random":
        return generate_random_dag(
            num_tasks=num_tasks,
            edge_prob=edge_prob,
            num_processors=num_processors,
            seed=seed
        )

    if graph_family == "erdos_renyi":
        return generate_erdos_renyi_dag(
            num_tasks=num_tasks,
            p=graph_params["p"],
            num_processors=num_processors,
            seed=seed
        )

    if graph_family == "barabasi_albert":
        return generate_barabasi_albert_dag(
            num_tasks=num_tasks,
            m=graph_params["m"],
            num_processors=num_processors,
            seed=seed
        )

    if graph_family == "watts_strogatz":
        return generate_watts_strogatz_dag(
            num_tasks=num_tasks,
            k=graph_params["k"],
            p=graph_params["p"],
            num_processors=num_processors,
            seed=seed
        )

    raise ValueError(f"Unsupported graph family: {graph_family}")


def run_batch_experiment(
    task_sizes=(20, 50, 100),
    processor_counts=(2, 4, 8),
    graphs_per_setting=50,
    train_ratio=0.8,
    edge_prob=0.2,
    n_trials=50,
):
    run_dir = create_run_dir("results")
    ablation_configs = get_ablation_configs()
    graph_family_configs = get_graph_family_configs(edge_prob)

    config = {
        "task_sizes": list(task_sizes),
        "processor_counts": list(processor_counts),
        "graphs_per_setting": graphs_per_setting,
        "train_ratio": train_ratio,
        "edge_prob": edge_prob,
        "n_trials": n_trials,
        "ablations": [a["name"] for a in ablation_configs],
        "graph_families": [g["name"] for g in graph_family_configs],
        "total_expected_dags": (
            len(task_sizes)
            * len(processor_counts)
            * graphs_per_setting
            * len(ablation_configs)
            * len(graph_family_configs)
        ),
    }
    save_config(run_dir, config)

    all_results = []

    for ablation in ablation_configs:
        print("\n" + "=" * 80)
        print(f"Running ablation: {ablation['name']}")
        print("=" * 80)

        for graph_cfg in graph_family_configs:
            graph_family = graph_cfg["name"]
            graph_params = graph_cfg["params"]

            print("\n" + "-" * 80)
            print(f"Graph family: {graph_family} | params={graph_params}")
            print("-" * 80)

            for num_tasks in task_sizes:
                for num_processors in processor_counts:
                    processors = list(range(num_processors))

                    print(
                        f"\nRunning setting: "
                        f"ablation={ablation['name']}, "
                        f"graph_family={graph_family}, "
                        f"tasks={num_tasks}, "
                        f"processors={num_processors}"
                    )

                    dags = []
                    for seed in range(graphs_per_setting):
                        dag = build_dag(
                            graph_family=graph_family,
                            graph_params=graph_params,
                            num_tasks=num_tasks,
                            num_processors=num_processors,
                            edge_prob=edge_prob,
                            seed=seed,
                        )
                        features = extract_features(dag, processors)
                        dags.append((seed, dag, features))

                    split_idx = int(train_ratio * graphs_per_setting)
                    train_items = dags[:split_idx]
                    test_items = dags[split_idx:]

                    # -----------------------------
                    # Feature selection / prompting
                    # -----------------------------
                    if ablation["use_llm"]:
                        prompt = build_prompt(
                            num_tasks=num_tasks,
                            num_processors=num_processors,
                            edge_prob=edge_prob,
                            num_train_graphs=len(train_items),
                            graph_family=graph_family,
                        )
                        llm_response = ask_llm(prompt)
                        feature_names = parse_feature_names(llm_response)

                        if not feature_names:
                            feature_names = ["rank_u", "rank_d", "fork", "comm_weight"]
                            llm_response = "LLM parsing failed. Falling back to default features."
                    else:
                        feature_names = ablation["fixed_feature_names"]
                        llm_response = f"Ablation {ablation['name']}: fixed handcrafted features used."

                    template = PriorityTemplate(feature_names)

                    # -----------------------------
                    # Train phase
                    # -----------------------------
                    train_dags = [(dag, features) for _, dag, features in train_items]

                    if ablation["use_bo"]:
                        best_theta, best_value = optimize_theta_on_dataset(
                            train_dags=train_dags,
                            processors=processors,
                            template=template,
                            n_trials=n_trials,
                        )
                    else:
                        best_theta = default_theta_for_template(template)
                        best_value = None

                    print(f"Template features: {template.feature_names}")
                    print(f"Best theta: {best_theta}")
                    if best_value is not None:
                        print(f"Best train avg makespan: {best_value}")
                    else:
                        print("BO skipped: using default theta")

                    # -----------------------------
                    # Test phase
                    # -----------------------------
                    for seed, dag, features in test_items:
                        result = evaluate_single_dag(
                            dag=dag,
                            processors=processors,
                            features=features,
                            template=template,
                            theta=best_theta,
                            llm_response=llm_response,
                        )

                        row = {
                            "ablation": ablation["name"],
                            "graph_family": graph_family,
                            "graph_params": str(graph_params),
                            "setting_tasks": num_tasks,
                            "setting_processors": num_processors,
                            "seed": seed,
                            "split": "test",
                            "HEFT": result["HEFT"],
                            "CPOP": result["CPOP"],
                            "RKHS_BO": result["RKHS_BO"],
                            "template_features": ",".join(result["template_features"]),
                            "theta": str(result["theta"]),
                            "llm_response": result["llm_response"].replace("\n", " "),
                        }
                        all_results.append(row)

                        print(
                            f"ablation={ablation['name']} | "
                            f"graph_family={graph_family} | "
                            f"test_seed={seed:02d} | "
                            f"HEFT={result['HEFT']} | "
                            f"CPOP={result['CPOP']} | "
                            f"RKHS_BO={result['RKHS_BO']}"
                        )

    save_per_dag_results(run_dir, all_results)

    summary = build_summary(all_results)
    save_summary_json(run_dir, summary)
    save_summary_txt(run_dir, config, summary)

    print("\nSaved outputs to:", run_dir)
    print("Files created:")
    print("- config.json")
    print("- per_dag_results.csv")
    print("- summary.json")
    print("- summary.txt")


def main():
    run_batch_experiment(
        task_sizes=(20, 50, 100),
        processor_counts=(2, 4, 8),
        graphs_per_setting=50,
        train_ratio=0.8,
        edge_prob=0.2,
        n_trials=50,
    )


if __name__ == "__main__":
    main()