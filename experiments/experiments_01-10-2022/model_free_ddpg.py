import gym
import tensorflow as tf
import numpy as np
from experiments.agent_experiments import experiment_model_free_ddpg

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

actor_params = {
    "observation_dim": 2,
    "action_max": 1,
    # "hidden_units": [16, 16]
}

critic_params = {
    "observation_dim": 2,
    "action_dim": 1,
    # "state_hidden_units": [16],
    # "action_hidden_units": [16],
    # "out_hidden_units": [20]
}

observation_max = np.array([0.6, 0.07])
observation_min = np.array([-1.2, -0.07])
observation_noise_stddev = [0, 0]

agent_time_ratio = 6

num_agents = 50

NORMAL_RUNS = 15
TRAN_RUNS = 0
HABIT_RUNS = 0
FLIP_DYNAMICS_RUNS = 0
EPISODES_BETWEEN_HABIT_TESTS = 10

experiment_name = "../../experiment_results/mf_ddpg_500_buffer"

env = gym.make('MountainCarContinuous-v0')

experiment_model_free_ddpg(experiment_name,
                           env,
                           observation_min,
                           observation_max,
                           observation_noise_stddev,
                           num_agents,
                           NORMAL_RUNS,
                           FLIP_DYNAMICS_RUNS,
                           EPISODES_BETWEEN_HABIT_TESTS,
                           actor_params,
                           critic_params,
                           ddpg_buffer_size=500,
                           ddpg_agent_time_ratio=6)
