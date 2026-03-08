def main():
    run_batch_experiment(
        task_sizes=(20,),
        processor_counts=(2,),
        graphs_per_setting=5,
        train_ratio=0.8,
        edge_prob=0.2,
        n_trials=5,
    )