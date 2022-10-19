from agent_and_models.vae import VAE, create_decoder, create_encoder
from agent_and_models.transition_gru import TransitionGRU
from agent_and_models.daif_agent import DAIFAgent
from agent_and_models.prior_preferences_model import PriorPreferencesModel
from agent_and_models.a2c import PolicyGradientNetwork, A2CAgent
from agent_and_models.ddpg import *

from util import test_policy, habit_policy
from train_agent import train_single_agent, train_single_model_free_agent
import pandas as pd


####################################################################
#       EXPERIMENTS WITH HABITUAL ACTION INCLUDED IN THE MODEL     #
####################################################################

def habit_action_A2C_experiment(
        experiment_save_path,
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
        habit_net = PolicyGradientNetwork(**a2c_params)
        habit_net.compile(optimizer=tf.keras.optimizers.Adam())

        # make the PRIOR NET
        prior_model = PriorPreferencesModel(**prior_params)

        # make the agent
        daifa = DAIFAgent(prior_model=prior_model, vae=vae, tran=tran, habitual_action_net=habit_net, **agent_params)

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
        daifa.use_habit_policy = True
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

    all_results.to_csv(f"{experiment_save_path}_agent_results.csv")
    all_habit_results.to_csv(f"{experiment_save_path}_habit_results.csv")

    print("EXPERIMENT FINISHED")


def habit_action_DDPG_experiment(
        experiment_save_path,
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
        agent_params,
        ddpg_buffer_size=1000):

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
        habit_net = BasicDDPG(actor_model, critic_model, target_actor, target_critic, tau=0.005, buffer_capacity=ddpg_buffer_size, critic_optimizer=critic_optimizer, actor_optimizer=actor_optimizer)

        # make the PRIOR NET
        prior_model = PriorPreferencesModel(**prior_params)

        # make the agent
        daifa = DAIFAgent(prior_model=prior_model, vae=vae, tran=tran, habitual_action_net=habit_net, **agent_params)

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
        daifa.use_habit_policy = True
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

    all_results.to_csv(f"{experiment_save_path}_agent_results.csv")
    all_habit_results.to_csv(f"{experiment_save_path}_habit_results.csv")

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
        daifa = DAIFAgent(vae=vae, tran=tran, **agent_params)

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

    all_results.to_csv(f"{experiment_name}_agent_results.csv")

    print("EXPERIMENT FINISHED")


####################################################################
#                BASIC EXPERIMENT WITH PRIOR MODEL                 #
####################################################################

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
        prior_model = PriorPreferencesModel(**prior_params)

        # make the agent
        daifa = DAIFAgent(prior_model=prior_model, vae=vae, tran=tran, **agent_params)

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

    all_results.to_csv(f"{experiment_name}_agent_results.csv")

    print("EXPERIMENT FINISHED")


####################################################################
#              BASIC EXPERIMENT WITH MODEL-FREE DDPG               #
####################################################################

def experiment_model_free_ddpg(
        experiment_name,
        env,
        observation_min,
        observation_max,
        observation_noise_stddev,
        num_agents,
        normal_runs,
        flip_dynamics_runs,
        episodes_between_habit_tests,
        actor_params,
        critic_params,
        ddpg_buffer_size,
        ddpg_agent_time_ratio):

    # track experiment results
    all_results = []
    all_habit_results = []

    for agent_num in range(num_agents):

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

        ou_noise = OUActionNoise(np.zeros(1), np.ones(1)*0.2)

        ddpg_agent = DDPGAgent(agent_time_ratio=ddpg_agent_time_ratio, ou_noise=ou_noise, actor=actor_model, critic=critic_model, target_actor=target_actor, target_critic=target_critic, tau=0.005, buffer_capacity=ddpg_buffer_size, critic_optimizer=critic_optimizer, actor_optimizer=actor_optimizer)

        # store and track results for this agent
        full_run_results = []
        full_run_habit_results = []
        habit_run_number = 0

        # Some epochs of VAE
        for n in range(normal_runs):

            ddpg_agent, results = train_single_model_free_agent(env, ddpg_agent, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False)
            full_run_results.append(results)

            p = ddpg_agent.select_action
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, ddpg_agent.agent_time_ratio)
            res["run_num"] = habit_run_number
            habit_run_number += 1
            full_run_habit_results.append(res)

        # Flip the dynamics
        for n in range(flip_dynamics_runs):

            ddpg_agent, results = train_single_model_free_agent(env, ddpg_agent, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False, flip_dynamics=True)
            full_run_results.append(results)

            p = ddpg_agent.select_action
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, ddpg_agent.agent_time_ratio)
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

    all_results.to_csv(f"{experiment_name}_agent_results.csv")
    all_habit_results.to_csv(f"{experiment_name}_habit_results.csv")

    print("EXPERIMENT FINISHED")


####################################################################
#              BASIC EXPERIMENT WITH MODEL-FREE A2C                #
####################################################################

def experiment_model_free_a2c(
        experiment_name,
        env,
        observation_min,
        observation_max,
        observation_noise_stddev,
        num_agents,
        normal_runs,
        flip_dynamics_runs,
        episodes_between_habit_tests,
        prior_params,
        a2c_params,
        a2c_agent_time_ratio):

    # track experiment results
    all_results = []
    all_habit_results = []

    for agent_num in range(num_agents):

        # # make the HABIT ACTION NET
        policy_net = PolicyGradientNetwork(**a2c_params)
        policy_net.compile(optimizer=tf.keras.optimizers.Adam())

        # make the PRIOR NET
        value_net = PriorPreferencesModel(**prior_params)

        # make the agent
        a2c_agent = A2CAgent(policy_net=policy_net, value_net=value_net, agent_time_ratio=a2c_agent_time_ratio)

        # store and track results for this agent
        full_run_results = []
        full_run_habit_results = []
        habit_run_number = 0

        # Some epochs of VAE
        for n in range(normal_runs):

            a2c_agent, results = train_single_model_free_agent(env, a2c_agent, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False)
            full_run_results.append(results)

            p = a2c_agent.policy_net
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, a2c_agent.agent_time_ratio)
            res["run_num"] = habit_run_number
            habit_run_number += 1
            full_run_habit_results.append(res)

        # Flip the dynamics
        for n in range(flip_dynamics_runs):

            a2c_agent, results = train_single_model_free_agent(env, a2c_agent, observation_max, observation_min, observation_noise_stddev, num_episodes=episodes_between_habit_tests, render_env=False, flip_dynamics=True)
            full_run_results.append(results)

            p = a2c_agent.policy_net
            res = test_policy(env, p, observation_max, observation_min, observation_noise_stddev, 20, a2c_agent.agent_time_ratio)
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

    all_results.to_csv(f"{experiment_name}_agent_results.csv")
    all_habit_results.to_csv(f"{experiment_name}_habit_results.csv")

    print("EXPERIMENT FINISHED")