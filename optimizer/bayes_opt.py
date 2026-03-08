import optuna
from core.scheduler import list_schedule

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_theta_on_dataset(train_dags, processors, template, n_trials=50):
    def objective(trial):
        theta = {
            name: trial.suggest_float(name, -3.0, 3.0)
            for name in template.feature_names
        }

        total_makespan = 0.0

        for dag, features in train_dags:
            scores = {
                v: template.score(features[v], theta)
                for v in dag.tasks
            }
            _, makespan = list_schedule(dag, processors, scores)
            total_makespan += makespan

        avg_makespan = total_makespan / len(train_dags)
        return avg_makespan

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value