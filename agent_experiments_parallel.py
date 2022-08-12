# agent imports
from recurrent_agent import DAIFAgentRecurrent
from vae_recurrent import VAE, create_encoder, create_decoder
from transition_gru import TransitionGRU
from prior_model import PriorModelBellman

# gym imports and training
import gym
from train_agent import train_agent

# utility
from util import transform_observations
import numpy as np
import pandas as pd
import multiprocessing
import tensorflow as tf


def experiment_feef_no_prior(agent_id, return_dict):

    print("Agent", agent_id)

    enc = create_encoder(2, 2, [20])
    dec = create_decoder(2, 2, [20])
    vae = VAE(enc, dec, [0, 0], [0.3, 0.3], llik_scaling=10000)

    tran = TransitionGRU(2, 1, 12, 60, 2)

    # unscaled prior mean and prior stddev
    prior_mean = [0.45, 0]
    prior_stddev = [1, 1]

    observation_max = np.array([0.6, 0.07])
    observation_min = np.array([-1.2, -0.07])

    observation_noise_stddev = [0, 0]

    scaled_prior_mean = transform_observations(prior_mean, observation_max, observation_min, [0, 0])  # no noise on prior

    daifa = DAIFAgentRecurrent(None,
                               vae,
                               tran,
                               scaled_prior_mean,
                               prior_stddev,
                               planning_horizon=15,
                               use_kl_extrinsic=True,
                               use_kl_intrinsic=True,
                               use_FEEF=True,
                               vae_train_epochs=1,
                               tran_train_epochs=1)

    # train the agent on the env
    env = gym.make('MountainCarContinuous-v0')
    agent, results = train_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=10, action_repeats=6, num_actions_to_execute=2)

    # add a column with agent id
    results["agent_num"] = agent_id

    return_dict[agent_id] = results


def experiment_feef_with_prior_model(agent_id, return_dict):

    print("Agent", agent_id)

    enc = create_encoder(2, 2, [20])
    dec = create_decoder(2, 2, [20])
    vae = VAE(enc, dec, [0, 0], [0.3, 0.3], llik_scaling=1)

    tran = TransitionGRU(2, 1, 12, 20, 2)

    # unscaled prior mean and prior stddev
    prior_model = PriorModelBellman(2)

    observation_max = np.array([0.6, 0.07])
    observation_min = np.array([-1.2, -0.07])

    # observation_noise_stddev = [0, 0]
    observation_noise_stddev = [0.05, 0.05]

    daifa = DAIFAgentRecurrent(prior_model,
                               vae,
                               tran,
                               None,
                               None,
                               train_prior_model=True,
                               planning_horizon=5,
                               use_kl_extrinsic=True,
                               use_kl_intrinsic=True,
                               use_FEEF=True,
                               vae_train_epochs=1,
                               tran_train_epochs=1)

    # train the agent on the env
    env = gym.make('MountainCarContinuous-v0')
    agent, results = train_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=40, action_repeats=6, num_actions_to_execute=2)

    # add a column with agent id
    results["agent_num"] = agent_id

    return_dict[agent_id] = results


def experiment_efe_no_prior_model(agent_id, return_dict):

    print("Agent", agent_id)

    enc = create_encoder(2, 2, [20])
    dec = create_decoder(2, 2, [20])
    vae = VAE(enc, dec, [0, 0], [0.3, 0.3], llik_scaling=10000)

    tran = TransitionGRU(2, 1, 12, 60, 2)

    # unscaled prior mean and prior stddev
    prior_mean = [0.45, 0]
    prior_stddev = [1, 1]

    observation_max = np.array([0.6, 0.07])
    observation_min = np.array([-1.2, -0.07])

    observation_noise_stddev = [0, 0]

    scaled_prior_mean = transform_observations(prior_mean, observation_max, observation_min, [0,0])  # no noise on prior

    daifa = DAIFAgentRecurrent(None,
                               vae,
                               tran,
                               scaled_prior_mean,
                               prior_stddev,
                               planning_horizon=15,
                               use_kl_extrinsic=True,
                               use_kl_intrinsic=True,
                               use_FEEF=False,
                               vae_train_epochs=1,
                               tran_train_epochs=1)

    # train the agent on the env
    env = gym.make('MountainCarContinuous-v0')
    agent, results = train_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=10, action_repeats=6, num_actions_to_execute=2)

    # add a column with agent id
    results["agent_num"] = agent_id

    # add results to be returned
    return_dict[agent_id] = results


def run_parallel_experiment(experiment_function_name, num_agents_to_train, output_directory_path, pandas_mode="a", use_cpu=True):

    if use_cpu:
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = []
    for i in range(num_agents_to_train):
        p = multiprocessing.Process(target=experiment_function_name, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    results_data_frames = return_dict.values()
    full_results = pd.concat(results_data_frames)

    full_results.reset_index(drop=True)
    full_results.to_csv(output_directory_path, index=False, mode=pandas_mode, header=False)

    print("EXPERIMENT COMPLETE")





