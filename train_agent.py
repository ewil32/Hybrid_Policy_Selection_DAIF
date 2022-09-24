import numpy as np
# import gym
import pandas as pd
import tensorflow as tf

from util import transform_observations, transform_image


def train_single_agent(mcc_env,
                       agent,
                       obs_max,
                       obs_min,
                       observation_noise_stddev,
                       num_episodes=100,
                       render_env=False,
                       flip_dynamics=False):

    # Set up to store results in pandas frame
    cols = ["episode", "success", "sim_steps", "VFE_post_run", "noise_stddev", "percent_use_fast_thinking", "total_reward", "agent_time_ratio"]
    rows = []

    for n in range(num_episodes):

        print("Episode", n+1)

        # reset the agent
        agent.reset_all_states()

        # get the first observation from the environment
        first_observation = mcc_env.reset()
        print(first_observation)

        # apply noise to and scaling to first observation
        observation_noisy = transform_observations(first_observation, obs_max, obs_min, observation_noise_stddev)
        observation_noisy = observation_noisy.reshape(1, observation_noisy.shape[0])
        # loop until episode ends or the agent succeeds
        t = 0
        reward = None
        done = False
        while not done:

            if render_env:
                mcc_env.render()

            action = agent.perceive_and_act(observation_noisy, reward=reward, done=done)
            if flip_dynamics:
                action = np.array(action) * -1

            observation, reward, done, info = mcc_env.step(action)  # action should be array to satisfy gym requirements
            observation_noisy = transform_observations(observation, obs_max, obs_min, observation_noise_stddev)
            observation_noisy = observation_noisy.reshape(1, observation_noisy.shape[0])

            t += 1

        # final training when the episode is done
        agent.perceive_and_act(observation_noisy, reward=reward, done=done)

        success = t < 999

        # get the VFE of the model for the run
        VFE = float(tf.reduce_mean(agent.model_vae.compute_loss(np.vstack(agent.full_observation_sequence))))
        percent_use_fast_thinking = agent.num_fast_thinking_choices / len(agent.full_action_sequence)
        total_reward = np.sum(agent.full_reward_sequence)

        rows.append(dict(zip(cols, [n, success, t, VFE, observation_noise_stddev, percent_use_fast_thinking, total_reward, agent.agent_time_ratio])))

        if success:
            print(f"Success in episode {n+1} at time step {t} with reward {total_reward}")
        else:
            print("No Success")

    results = pd.DataFrame(rows, columns=cols)

    mcc_env.close()

    return agent, results


def train_single_agent_car_racing(cr_env,
                                  agent,
                                  num_episodes=100,
                                  render_env=False):

    # Set up to store results in pandas frame
    cols = ["episode", "success", "sim_steps", "VFE_post_run", "noise_stddev"]
    rows = []

    for n in range(num_episodes):

        print("Episode", n+1)

        # reset the agent
        agent.reset_all_states()

        # get the first observation from the environment
        first_observation = cr_env.reset()

        # apply noise to and scaling to first observation
        scaled_observation = transform_image(first_observation, 16, 80, 16, 80)
        scaled_observation = scaled_observation.reshape([1] + list(scaled_observation.shape))

        # loop until episode ends or the agent succeeds
        t = 0
        reward = None
        total_reward = 0
        done = False
        while not done:

            if render_env:
                cr_env.render()

            action = agent.perceive_and_act(scaled_observation, reward=reward, done=done)

            turn_direction = action[0]
            brake_or_gas = action[1]

            # make action into form that works for this
            if brake_or_gas >= 0:
                gas = brake_or_gas
                action_to_execute = np.array([turn_direction, gas, 0])
            else:
                brake = -1*brake_or_gas
                action_to_execute = np.array([turn_direction, 0, brake])

            observation, reward, done, info, *rest = cr_env.step(action_to_execute)  # action should be array to satisfy gym requirements
            # print(observation.shape)
            scaled_observation = transform_image(observation, 16, 80, 16, 80)
            scaled_observation = scaled_observation.reshape([1] + list(scaled_observation.shape))

            t += 1
            total_reward += reward

            if total_reward < -100:
                done = True

        # print(scaled_observation.shape)
        # final training when the episode is done
        agent.perceive_and_act(scaled_observation, reward=reward, done=done)

        success = t < 999

        # get the VFE of the model for the run
        # VFE = float(tf.reduce_mean(agent.model_vae.compute_loss(all_post_observations)))
        VFE = 0

        rows.append(dict(zip(cols, [n, success, t, VFE, 0])))

        if success:
            print("Success in episode", n+1, "at time step", t)
        else:
            print("No Success")

    results = pd.DataFrame(rows, columns=cols)

    cr_env.close()

    return agent, results
