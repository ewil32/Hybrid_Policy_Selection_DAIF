import gym

from vae_recurrent import VAE, create_decoder, create_encoder
from transition_gru import TransitionGRU
from recurrent_agent import DAIFAgentRecurrent
from prior_model import PriorModelBellman
from habitual_action_network import HabitualAction, compute_discounted_cumulative_reward
from ddpg import *

from util import random_observation_sequence, transform_observations, test_policy, habit_policy
from train_agent import train_single_agent
import pandas as pd

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

pln_hrzn = 5
latent_dim = 2
obs_dim = 2
num_agents = 10

VAE_RUNS = 6
TRAN_RUNS = 2
HABIT_RUNS = 2
FLIP_DYNAMICS_RUNS = 4
EPISODES_BETWEEN_HABIT_TESTS = 10

experiment_name = "A2C_train_from_start"

all_results = []
all_habit_results = []

# train the agent on the env
env = gym.make('MountainCarContinuous-v0')

for agent_num in range(num_agents):

    # make the VAE
    enc = create_encoder(2, latent_dim, [20])
    dec = create_decoder(latent_dim, 2, [20])
    vae = VAE(enc, dec, latent_dim,  [0]*latent_dim, [0.3]*latent_dim, train_epochs=2, show_training=False)
    vae.compile(optimizer=tf.keras.optimizers.Adam())

    # make the TRANSITION
    tran = TransitionGRU(latent_dim, 1, 2*pln_hrzn*latent_dim, 2, train_epochs=2, show_training=False)
    tran.compile(optimizer=tf.keras.optimizers.Adam())

    # # make the HABIT ACTION NET
    habit_net = HabitualAction(latent_dim, 1, [16, 16], train_epochs=2, show_training=False)
    habit_net.compile(optimizer=tf.keras.optimizers.Adam())

    # make the PRIOR NET
    prior_model = PriorModelBellman(latent_dim, output_dim=1, show_training=False, use_tanh_on_output=False)

    observation_max = np.array([0.6, 0.07])
    observation_min = np.array([-1.2, -0.07])

    # observation_noise_stddev = [0, 0]
    observation_noise_stddev = [0.05, 0.05]

    # # unscaled prior mean and prior stddev
    # prior_mean = [0.45, 0]
    # prior_stddev = [1, 1]
    # scaled_prior_mean = transform_observations(prior_mean, observation_max, observation_min, [0,0])  # no noise on prior

    daifa = DAIFAgentRecurrent(prior_model,
                               vae,
                               tran,
                               habit_net,
                               planning_horizon=pln_hrzn,
                               use_kl_extrinsic=True,  # maybe this works
                               use_kl_intrinsic=True,
                               use_FEEF=False,
                               train_habit_net=True,
                               train_prior_model=True,
                               train_tran=True,
                               train_after_exploring=True,
                               train_with_replay=True,
                               use_fast_thinking=True,
                               habit_model_type="PG",
                               uncertainty_tolerance=0.1)

    full_run_results = []
    full_run_habit_results = []
    habit_run_number = 0

    # Some epcohs of VAE
    for n in range(VAE_RUNS):

        daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=EPISODES_BETWEEN_HABIT_TESTS, render_env=False)
        full_run_results.append(results)

        p = habit_policy(daifa)
        res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
        res["run_num"] = habit_run_number
        habit_run_number += 1
        full_run_habit_results.append(res)

    daifa.train_vae = False
    daifa.model_vae.show_training = False

    # Stop training VAE and fine tune tran
    for n in range(TRAN_RUNS):

        daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=EPISODES_BETWEEN_HABIT_TESTS, render_env=False)
        full_run_results.append(results)

        p = habit_policy(daifa)
        res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
        res["run_num"] = habit_run_number
        habit_run_number += 1
        full_run_habit_results.append(res)

    # Switch on HABIT if it wasn't already on
    daifa.habit_action_model.show_training = False
    daifa.train_habit_net = True
    daifa.train_after_exploring = True
    daifa.use_fast_thinking = True
    daifa.uncertainty_tolerance = 0.1

    # Stop training VAE and fine tune tran
    for n in range(HABIT_RUNS):

        daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=EPISODES_BETWEEN_HABIT_TESTS, render_env=False)
        full_run_results.append(results)

        p = habit_policy(daifa)
        res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
        res["run_num"] = habit_run_number
        habit_run_number += 1
        full_run_habit_results.append(res)


    # Flip the dynamics
    for n in range(FLIP_DYNAMICS_RUNS):

        daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=EPISODES_BETWEEN_HABIT_TESTS, render_env=False, flip_dynamics=True)
        full_run_results.append(results)

        p = habit_policy(daifa)
        res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
        res["run_num"] = habit_run_number
        habit_run_number += 1
        full_run_habit_results.append(res)

    # collect the results for this agent
    full_run_results = pd.concat(full_run_results)
    full_run_results = full_run_results.reset_index(drop=True)
    full_run_results["episode"] = full_run_results.index
    full_run_results["agent_id"] = agent_num

    full_run_habit_results = pd.concat(full_run_habit_results)
    full_run_habit_results["agent_id"] = agent_num

    # add the results to all the agents
    all_results.append(full_run_results)
    all_habit_results.append(full_run_habit_results)


all_results = pd.concat(all_results)
all_results = all_results.reset_index(drop=True)
all_habit_results = pd.concat(all_habit_results)
all_habit_results = all_habit_results.reset_index(drop=True)

all_results.to_csv(f"../experiment_results/{experiment_name}_agent_results.csv")
all_habit_results.to_csv(f"../experiment_results/{experiment_name}_habit_results.csv")

print("EXPERIMENT FINISHED")
