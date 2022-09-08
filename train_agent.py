import numpy as np
# import gym
import pandas as pd
import tensorflow as tf

from util import transform_observations


def train_single_agent(mcc_env,
                       agent,
                       obs_max,
                       obs_min,
                       observation_noise_stddev,
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
            # print(action)
            observation, reward, done, info = mcc_env.step(action)  # action should be array to satisfy gym requirements
            # print(observation)
            observation_noisy = transform_observations(observation, obs_max, obs_min, observation_noise_stddev)
            observation_noisy = observation_noisy.reshape(1, observation_noisy.shape[0])

            t += 1

        # final training when the episode is done
        agent.perceive_and_act(observation_noisy, reward=reward, done=done)

        success = t < 999

        # get the VFE of the model for the run
        # VFE = float(tf.reduce_mean(agent.model_vae.compute_loss(all_post_observations)))
        VFE = 0

        rows.append(dict(zip(cols, [n, success, t, VFE, observation_noise_stddev])))

        if success:
            print("Success in episode", n+1, "at time step", t)
        else:
            print("No Success")

    results = pd.DataFrame(rows, columns=cols)

    mcc_env.close()

    return agent, results







