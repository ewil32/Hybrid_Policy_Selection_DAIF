from vae_recurrent import VAE, create_decoder, create_encoder
from transition_gru import TransitionGRU
from recurrent_agent import DAIFAgentRecurrent
from prior_model import PriorModelBellman
from habitual_action_network import HabitualAction, compute_discounted_cumulative_reward
from ddpg import *

from util import random_observation_sequence, transform_observations, test_policy, habit_policy
from train_agent import train_single_agent
import pandas as pd


####################################################################
#       EXPERIMENTS WITH HABITUAL ACTION INCLUDED IN THE MODEL     #
####################################################################

def habit_action_A2C_experiment(
        experiment_name,
        env,
        observation_min,
        observation_max,
        observation_noise_stddev,
        num_agents,
        vae_runs,
        tran_runs,
        habit_runs,
        flip_dynamics_runs,
        episodes_between_habit_tests,
        encoder_params,
        decoder_params,
        vae_params,
        tran_params,
        prior_params,
        a2c_params,
        agent_params):

    # track experiment results
    all_results = []
    all_habit_results = []

    for agent_num in range(num_agents):

        # make the VAE
        enc = create_encoder(**encoder_params)
        dec = create_decoder(**decoder_params)
        vae = VAE(enc, dec, **vae_params)
        vae.compile(optimizer=tf.keras.optimizers.Adam())

        # make the TRANSITION
        tran = TransitionGRU(**tran_params)
        tran.compile(optimizer=tf.keras.optimizers.Adam())

        # # make the HABIT ACTION NET
        habit_net = HabitualAction(**a2c_params)
        habit_net.compile(optimizer=tf.keras.optimizers.Adam())

        # make the PRIOR NET
        prior_model = PriorModelBellman(**prior_params)

        # make the agent
        daifa = DAIFAgentRecurrent(prior_model=prior_model, vae=vae, tran=tran, habitual_action_net=habit_net, **agent_params)

        # store and track results for this agent
        full_run_results = []
        full_run_habit_results = []
        habit_run_number = 0

        # Some epochs of VAE
        for n in range(vae_runs):

            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False)
            full_run_results.append(results)

            p = habit_policy(daifa)
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
            res["run_num"] = habit_run_number
            habit_run_number += 1
            full_run_habit_results.append(res)

        # Stop training VAE and keep training tran
        daifa.train_vae = False
        daifa.model_vae.show_training = False
        for n in range(tran_runs):

            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False)
            full_run_results.append(results)

            p = habit_policy(daifa)
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
            res["run_num"] = habit_run_number
            habit_run_number += 1
            full_run_habit_results.append(res)

        # Switch on HABIT if it wasn't already on
        daifa.habit_action_model.show_training = False
        daifa.train_habit_net = True
        daifa.train_during_episode = True
        daifa.use_fast_thinking = True
        daifa.uncertainty_tolerance = agent_params["uncertainty_tolerance"]

        # Train habit
        for n in range(habit_runs):

            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False)
            full_run_results.append(results)

            p = habit_policy(daifa)
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
            res["run_num"] = habit_run_number
            habit_run_number += 1
            full_run_habit_results.append(res)

        # Flip the dynamics
        for n in range(flip_dynamics_runs):

            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False, flip_dynamics=True)
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

    # write the final results to csv
    all_results = pd.concat(all_results)
    all_results = all_results.reset_index(drop=True)
    all_habit_results = pd.concat(all_habit_results)
    all_habit_results = all_habit_results.reset_index(drop=True)

    all_results.to_csv(f"../experiment_results/{experiment_name}_agent_results.csv")
    all_habit_results.to_csv(f"../experiment_results/{experiment_name}_habit_results.csv")

    print("EXPERIMENT FINISHED")



def habit_action_DDPG_experiment(
        experiment_name,
        env,
        observation_min,
        observation_max,
        observation_noise_stddev,
        num_agents,
        vae_runs,
        tran_runs,
        habit_runs,
        flip_dynamics_runs,
        episodes_between_habit_tests,
        encoder_params,
        decoder_params,
        vae_params,
        tran_params,
        prior_params,
        actor_params,
        critic_params,
        agent_params):

    # track experiment results
    all_results = []
    all_habit_results = []

    for agent_num in range(num_agents):

        # make the VAE
        enc = create_encoder(**encoder_params)
        dec = create_decoder(**decoder_params)
        vae = VAE(enc, dec, **vae_params)
        vae.compile(optimizer=tf.keras.optimizers.Adam())

        # make the TRANSITION
        tran = TransitionGRU(**tran_params)
        tran.compile(optimizer=tf.keras.optimizers.Adam())

        # # make the HABIT ACTION NET
        actor_model = get_actor(**actor_params)
        critic_model = get_critic(**critic_params)
        target_actor = get_actor(**actor_params)
        target_critic = get_critic(**critic_params)

        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())
        critic_optimizer = tf.keras.optimizers.Adam(0.0001)
        actor_optimizer = tf.keras.optimizers.Adam(0.00005)
        habit_net = BasicDDPG(actor_model, critic_model, target_actor, target_critic, tau=0.005, critic_optimizer=critic_optimizer, actor_optimizer=actor_optimizer)

        # make the PRIOR NET
        prior_model = PriorModelBellman(**prior_params)

        # make the agent
        daifa = DAIFAgentRecurrent(prior_model=prior_model, vae=vae, tran=tran, habitual_action_net=habit_net, **agent_params)

        # store and track results for this agent
        full_run_results = []
        full_run_habit_results = []
        habit_run_number = 0

        # Some epochs of VAE
        for n in range(vae_runs):

            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False)
            full_run_results.append(results)

            p = habit_policy(daifa)
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
            res["run_num"] = habit_run_number
            habit_run_number += 1
            full_run_habit_results.append(res)

        # Stop training VAE and keep training tran
        daifa.train_vae = False
        daifa.model_vae.show_training = False
        for n in range(tran_runs):

            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False)
            full_run_results.append(results)

            p = habit_policy(daifa)
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
            res["run_num"] = habit_run_number
            habit_run_number += 1
            full_run_habit_results.append(res)

        # Switch on HABIT if it wasn't already on
        daifa.habit_action_model.show_training = False
        daifa.train_habit_net = True
        daifa.train_during_episode = True
        daifa.use_fast_thinking = True
        daifa.uncertainty_tolerance = agent_params["uncertainty_tolerance"]

        # Stop training VAE and fine tune tran
        for n in range(habit_runs):

            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False)
            full_run_results.append(results)

            p = habit_policy(daifa)
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, daifa.agent_time_ratio)
            res["run_num"] = habit_run_number
            habit_run_number += 1
            full_run_habit_results.append(res)

        # Flip the dynamics
        for n in range(flip_dynamics_runs):

            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False, flip_dynamics=True)
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

    # write the final results to csv
    all_results = pd.concat(all_results)
    all_results = all_results.reset_index(drop=True)
    all_habit_results = pd.concat(all_habit_results)
    all_habit_results = all_habit_results.reset_index(drop=True)

    all_results.to_csv(f"../experiment_results/{experiment_name}_agent_results.csv")
    all_habit_results.to_csv(f"../experiment_results/{experiment_name}_habit_results.csv")

    print("EXPERIMENT FINISHED")


####################################################################
#                BASIC EXPERIMENT WITHOUT PRIOR MODEL              #
####################################################################

def basic_experiment(
        experiment_name,
        env,
        observation_min,
        observation_max,
        observation_noise_stddev,
        num_agents,
        vae_runs,
        tran_runs,
        flip_dynamics_runs,
        encoder_params,
        decoder_params,
        vae_params,
        tran_params,
        agent_params):

    # track experiment results
    all_results = []

    for agent_num in range(num_agents):

        # make the VAE
        enc = create_encoder(**encoder_params)
        dec = create_decoder(**decoder_params)
        vae = VAE(enc, dec, **vae_params)
        vae.compile(optimizer=tf.keras.optimizers.Adam())

        # make the TRANSITION
        tran = TransitionGRU(**tran_params)
        tran.compile(optimizer=tf.keras.optimizers.Adam())

        # make the agent
        daifa = DAIFAgentRecurrent(vae=vae, tran=tran, **agent_params)

        # store and track results for this agent
        full_run_results = []

        # Some epochs of VAE
        if vae_runs > 0:
            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=vae_runs, render_env=False)
            full_run_results.append(results)

        # Stop training VAE and keep training tran
        daifa.train_vae = False
        daifa.model_vae.show_training = False

        if tran_runs > 0:
            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=tran_runs, render_env=False)
            full_run_results.append(results)

        # Flip the dynamics
        if flip_dynamics_runs > 0:
            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=flip_dynamics_runs, render_env=False, flip_dynamics=True)
            full_run_results.append(results)

        # collect the results for this agent
        full_run_results = pd.concat(full_run_results)
        full_run_results = full_run_results.reset_index(drop=True)
        full_run_results["episode"] = full_run_results.index
        full_run_results["agent_id"] = agent_num

        # add the results to all the agents
        all_results.append(full_run_results)

    # write the final results to csv
    all_results = pd.concat(all_results)
    all_results = all_results.reset_index(drop=True)

    all_results.to_csv(f"../experiment_results/{experiment_name}_agent_results.csv")

    print("EXPERIMENT FINISHED")


def experiment_with_prior_model(
        experiment_name,
        env,
        observation_min,
        observation_max,
        observation_noise_stddev,
        num_agents,
        vae_runs,
        tran_runs,
        flip_dynamics_runs,
        encoder_params,
        decoder_params,
        vae_params,
        tran_params,
        prior_params,
        agent_params):

    # track experiment results
    all_results = []

    for agent_num in range(num_agents):

        # make the VAE
        enc = create_encoder(**encoder_params)
        dec = create_decoder(**decoder_params)
        vae = VAE(enc, dec, **vae_params)
        vae.compile(optimizer=tf.keras.optimizers.Adam())

        # make the TRANSITION
        tran = TransitionGRU(**tran_params)
        tran.compile(optimizer=tf.keras.optimizers.Adam())

        # make the PRIOR NET
        prior_model = PriorModelBellman(**prior_params)

        # make the agent
        daifa = DAIFAgentRecurrent(prior_model=prior_model, vae=vae, tran=tran, **agent_params)

        # store and track results for this agent
        full_run_results = []

        # Some epochs of VAE
        if vae_runs > 0:
            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=vae_runs, render_env=False)
            full_run_results.append(results)

        # Stop training VAE and keep training tran
        daifa.train_vae = False
        daifa.model_vae.show_training = False

        if tran_runs > 0:
            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=tran_runs, render_env=False)
            full_run_results.append(results)

        # Flip the dynamics
        if flip_dynamics_runs > 0:
            daifa, results = train_single_agent(env, daifa, observation_max, observation_min, observation_noise_stddev, num_episodes=flip_dynamics_runs, render_env=False, flip_dynamics=True)
            full_run_results.append(results)

        # collect the results for this agent
        full_run_results = pd.concat(full_run_results)
        full_run_results = full_run_results.reset_index(drop=True)
        full_run_results["episode"] = full_run_results.index
        full_run_results["agent_id"] = agent_num

        # add the results to all the agents
        all_results.append(full_run_results)

    # write the final results to csv
    all_results = pd.concat(all_results)
    all_results = all_results.reset_index(drop=True)

    all_results.to_csv(f"../experiment_results/{experiment_name}_agent_results.csv")

    print("EXPERIMENT FINISHED")