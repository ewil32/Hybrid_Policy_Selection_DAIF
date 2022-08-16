from agent_experiments_parallel import *

if __name__ == "__main__":

    run_parallel_experiment(experiment_efe_no_prior_model, 10, "./experiments/efe_no_prior_model.csv")
    # run_parallel_experiment(experiment_feef_no_prior, 10, "./experiments/feef_no_prior.csv")
    # run_parallel_experiment(experiment_feef_with_prior_model, 10, "./experiments/feef_with_prior_model.csv")

    # run_parallel_experiment(experiment_feef_with_prior_model_no_replay, 10, "./experiments/feef_with_prior_no_replay.csv")

