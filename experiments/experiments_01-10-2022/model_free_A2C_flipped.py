import gym
import tensorflow as tf
import numpy as np
from experiments.agent_experiments import experiment_model_free_a2c

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

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


observation_max = np.array([0.6, 0.07])
observation_min = np.array([-1.2, -0.07])
observation_noise_stddev = [0, 0]

num_agents = 50

NORMAL_RUNS = 10
FLIP_DYNAMICS_RUNS = 5
EPISODES_BETWEEN_HABIT_TESTS = 10

experiment_name = "../../experiment_results/mf_A2C_flipped"

# train the agent on the env
env = gym.make('MountainCarContinuous-v0')

experiment_model_free_a2c(
    experiment_name,
    env,
    observation_min,
    observation_max,
    observation_noise_stddev,
    num_agents,
    NORMAL_RUNS,
    FLIP_DYNAMICS_RUNS,
    EPISODES_BETWEEN_HABIT_TESTS,
    prior_params,
    a2c_model_params,
    a2c_agent_time_ratio=6)
