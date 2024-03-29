import gym
import tensorflow as tf
import numpy as np
from experiments.agent_experiments import habit_action_A2C_experiment

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

encoder_params = {
    "input_dim": 2,
    "latent_dim": 2,
    "hidden_units": [20]
}

decoder_params = {
    "output_dim": 2,
    "latent_dim": 2,
    "hidden_units": [20]
}

vae_params = {
    "latent_dim": 2,
    "reg_mean": [0, 0],
    "reg_stddev": [0.3, 0.3],
    "recon_stddev": 0.05,
    "train_epochs": 2,
    "show_training": False
}

tran_params = {
    "latent_dim": 2,
    "action_dim": 1,
    "hidden_units": 20,
    "output_dim": 2,
    "train_epochs": 2,
    "show_training": False
}

prior_params = {
    "observation_dim": 2,
    "output_dim": 1,
    "iterate_train": 1,
    "discount_factor": 0.99,
    "training_epochs": 1,
    "show_training": False,
    "use_tanh_on_output": False
}

a2c_model_params = {
    "latent_dim": 2,
    "action_dim": 1,
    "dense_units": [16, 16],
    "action_std_dev": 0.05,
    "train_epochs": 2,
    "show_training": False,
    "discount_factor": 0.99
}

agent_params = {
    "given_prior_mean": None,
    "given_prior_stddev": None,
    "agent_time_ratio": 6,
    "actions_to_execute_when_exploring": 2,
    "planning_horizon": 5,
    "n_policies": 1500,
    "n_cem_policy_iterations": 2,
    "n_policy_candidates": 70,
    "train_vae": True,
    'train_tran': True,
    "train_prior_model": True,
    "train_habit_net": True,
    "train_with_replay": True,
    "train_during_episode": True,
    "use_efe_extrinsic": True,
    "use_kl_intrinsic": True,
    "use_FEEF": False,
    "use_habit_policy": True,
    "uncertainty_tolerance": 0.1,
    "habit_model_type": "A2C",
    "min_rewards_needed_to_train_prior": -10,
    "prior_model_scaling_factor": 1
}

observation_max = np.array([0.6, 0.07])
observation_min = np.array([-1.2, -0.07])
observation_noise_stddev = [0.05, 0.05]

num_agents = 50

VAE_RUNS = 6
TRAN_RUNS = 2
HABIT_RUNS = 2
FLIP_DYNAMICS_RUNS = 5
EPISODES_BETWEEN_HABIT_TESTS = 10

experiment_name = "../../experiment_results/A2C_from_start_flipped"

# train the agent on the env
env = gym.make('MountainCarContinuous-v0')

habit_action_A2C_experiment(experiment_name,
                            env,
                            observation_min,
                            observation_max,
                            observation_noise_stddev,
                            num_agents,
                            VAE_RUNS,
                            TRAN_RUNS,
                            HABIT_RUNS,
                            FLIP_DYNAMICS_RUNS,
                            EPISODES_BETWEEN_HABIT_TESTS,
                            encoder_params,
                            decoder_params,
                            vae_params,
                            tran_params,
                            prior_params,
                            a2c_model_params,
                            agent_params)

