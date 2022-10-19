import gym
import tensorflow as tf
import numpy as np
from experiments.agent_experiments import basic_experiment
from util import transform_observations

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

observation_max = np.array([0.6, 0.07])
observation_min = np.array([-1.2, -0.07])
observation_noise_stddev = [0.05, 0.05]

# unscaled prior mean and prior stddev
prior_mean = [0.45, 0]
prior_stddev = [1, 1]
scaled_prior_mean = transform_observations(prior_mean, observation_max, observation_min, [0, 0])  # no noise on prior

agent_params = {
    "prior_model": None,
    "habitual_action_net": None,
    "given_prior_mean": scaled_prior_mean,
    "given_prior_stddev": prior_stddev,
    "agent_time_ratio": 6,
    "actions_to_execute_when_exploring": 2,
    "planning_horizon": 5,
    "n_policies": 1500,
    "n_cem_policy_iterations": 2,
    "n_policy_candidates": 70,
    "train_vae": True,
    'train_tran': True,
    "train_prior_model": False,
    "train_habit_net": False,
    "train_with_replay": True,
    "train_during_episode": True,
    "use_efe_extrinsic": True,
    "use_kl_intrinsic": True,
    "use_FEEF": False,
    "use_habit_policy": False,
    "uncertainty_tolerance": 0.1,
    "habit_model_type": None
}

VAE_RUNS = 60
TRAN_RUNS = 90
FLIP_DYNAMICS_RUNS = 0
num_agents = 50

experiment_name = "../../experiment_results/VAE_halting_EFE"

# train the agent on the env
env = gym.make('MountainCarContinuous-v0')

basic_experiment(experiment_name,
                 env,
                 observation_min,
                 observation_max,
                 observation_noise_stddev,
                 num_agents,
                 VAE_RUNS,
                 TRAN_RUNS,
                 FLIP_DYNAMICS_RUNS,
                 encoder_params,
                 decoder_params,
                 vae_params,
                 tran_params,
                 agent_params)
