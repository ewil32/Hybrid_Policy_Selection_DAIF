import numpy as np
# import gym
import pandas as pd
import tensorflow as tf

from util import transform_observations


def run_episode(mcc_env, agent, obs_max, obs_min, observation_noise_stddev=[0.05, 0.05], action_repeats=6, num_actions_to_execute=2):

    # arrays to store observations, actions and rewards
    all_pre_observations = []
    all_post_observations = []
    all_action = []
    observation_sequence = []
    actions_executed = []
    agent_scale_observation_sequence = []
    reward_sequence = []

    # get the first observation from the environment
    first_observation, info = mcc_env.reset()
    first_observation = np.array([first_observation, 0])

    # apply noise to and scaling to first observation
    first_observation_noisy = transform_observations(first_observation, obs_max, obs_min, observation_noise_stddev)

    # find the first policy
    policy_observation = first_observation_noisy
    policy = agent.select_policy(policy_observation)

    # loop until episode ends or the agent succeeds
    t = 0
    while True:

        # if t % 10 == 0:
        #     print(t)

        # get the actions from the policy and reshape to desired form
        actions = policy.mean()
        actions = tf.reshape(actions, (actions.shape[0], agent.tran.action_dim))  # [num_actions, action_dim]
        actions = actions.numpy()

        # get the actions that we will execute before changing policy
        actions_to_execute = actions[0:num_actions_to_execute]

        # agent executes policy and gathers observations
        for action in actions_to_execute:
            for n in range(action_repeats):
                observation, reward, done, info = mcc_env.step(action)  # action should be array to satisfy gym requirements

                actions_executed.append(action)
                observation_sequence.append(observation)
                reward_sequence.append(reward)

                t += 1
                if done:
                    if t < 999:
                        print(policy_observation)
                        print(policy.mean())
                    return t < 999, agent, t, all_pre_observations, all_post_observations, all_action # the max for the environment

        actions_executed = np.array(actions_executed).reshape((len(actions_executed), agent.tran.action_dim))

        # scale and add noise to the observation
        observation_sequence = transform_observations(observation_sequence, obs_max, obs_min, observation_noise_stddev)

        # get the noisy observations for pre and post actions
        pre_observation_sequence = np.vstack([policy_observation, observation_sequence[:-1]])
        post_action_observation_sequence = observation_sequence

        # print(post_action_observation_sequence)

        all_pre_observations.append(pre_observation_sequence)
        all_post_observations.append(post_action_observation_sequence)
        all_action.append(actions_executed)

        # print("pol", policy_observation)
        # print("obs", observation_sequence)
        # print("pre", pre_observation_sequence)
        # print("post", post_action_observation_sequence)

        # if time to train the agent
        agent.train(pre_observation_sequence, post_action_observation_sequence, actions_executed, reward_sequence)

        # the new observation we use to select a policy is the last observation in observation_sequences
        policy_observation = observation_sequence[-1]

        # select a new policy and clear everything
        policy = agent.select_policy(policy_observation)

        # clear the observations
        observation_sequence = []
        reward_sequence = []
        actions_executed = []

    env.close()


def train_agent(mcc_env, agent, obs_max, obs_min, observation_noise_stddev, action_repeats, num_actions_to_execute, episode_length=1000, num_episodes=100):

    # Set up to store in pandas frame
    cols = ["episode", "success", "sim_steps", "noise_stddev"]
    rows = []

    for n in range(num_episodes):
        print("Episode", n+1)
        success, agent, t, *rest = run_episode(mcc_env, agent, obs_max, obs_min, observation_noise_stddev, action_repeats=action_repeats, num_actions_to_execute=num_actions_to_execute)

        rows.append(dict(zip(cols, [n, success, t, observation_noise_stddev])))

        if success:
            print("Success in episode", n+1, "at time step", t)
        else:
            print("No Success")

    results = pd.DataFrame(rows, columns=cols)

    return agent, results
