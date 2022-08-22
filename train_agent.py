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
                       action_repeats,
                       num_actions_to_execute,
                       num_episodes=100,
                       train_on_full_data=True,
                       show_replay_training=False,
                       replay_train_epochs=2,
                       train_during_episode=True,
                       train_vae=True,
                       train_tran=True,
                       train_prior=False,
                       train_habit=True):

    # Set up to store results in pandas frame
    cols = ["episode", "success", "sim_steps", "VFE_post_run", "noise_stddev"]
    rows = []

    for n in range(num_episodes):

        print("Episode", n+1)
        # arrays to store observations, actions and rewards

        all_pre_observations = []
        all_post_observations = []
        all_action = []
        observation_sequence = []
        actions_executed = []
        reward_sequence = []

        # get the first observation from the environment
        first_observation = mcc_env.reset()
        print(first_observation)
        mcc_env.render()
        # first_observation = np.array([first_observation, 0])

        # apply noise to and scaling to first observation
        first_observation_noisy = transform_observations(first_observation, obs_max, obs_min, observation_noise_stddev)

        # find the first policy
        policy_observation = first_observation_noisy
        policy = agent.select_policy(policy_observation)

        # loop until episode ends or the agent succeeds
        t = 0
        done = False
        while not done:

            # get the actions from the policy and reshape to desired form
            actions = policy.mean()
            actions = tf.reshape(actions, (actions.shape[0], agent.tran.action_dim))  # [num_actions, action_dim]
            actions = actions.numpy()

            # print(policy_observation, actions)

            # get the actions that we will execute before changing policy
            actions_to_execute = []
            for action in actions[0:num_actions_to_execute]:
                actions_to_execute = actions_to_execute + [action]*action_repeats

            # agent executes policy and gathers observations
            for action in actions_to_execute:
                observation, reward, done, info = mcc_env.step(action)  # action should be array to satisfy gym requirements

                # view the environment
                mcc_env.render()

                actions_executed.append(action)
                observation_sequence.append(observation)
                reward_sequence.append(reward)

                t += 1
                if done:

                    # did we succeed
                    if t < 999:
                        print(policy_observation)
                        print(policy.mean())

                        success = True

                    else:
                        success = False

                    # get a full observations set
                    all_post_observations = np.vstack(all_post_observations)
                    all_pre_observations = np.vstack(all_pre_observations)
                    all_action = np.vstack(all_action)

                    # should we train on final full data run
                    if train_on_full_data:
                        # agent.model_vae.fit(all_post_observations, epochs=replay_train_epochs, verbose=show_replay_training)
                        agent.reset_tran_hidden_state()
                        agent.train(all_pre_observations, all_post_observations, all_action, rewards=None, train_vae=train_vae, train_tran=train_tran, train_prior=train_prior, train_habit=train_habit)


                    # get the VFE of the model for the run
                    VFE = float(tf.reduce_mean(agent.model_vae.compute_loss(all_post_observations)))

                    # finally break free from the loop
                    break

            if not done:

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
                if train_during_episode:
                    agent.train(pre_observation_sequence, post_action_observation_sequence, actions_executed, reward_sequence, train_vae=train_vae, train_tran=train_tran, train_prior=train_prior, train_habit=train_habit)

                # the new observation we use to select a policy is the last observation in observation_sequences
                policy_observation = observation_sequence[-1]

                # select a new policy and clear everything
                policy = agent.select_policy(policy_observation)

                # clear the observations
                observation_sequence = []
                reward_sequence = []
                actions_executed = []

        rows.append(dict(zip(cols, [n, success, t, VFE, observation_noise_stddev])))

        if success:
            print("Success in episode", n+1, "at time step", t)
        else:
            print("No Success")

    results = pd.DataFrame(rows, columns=cols)

    mcc_env.close()

    return agent, results



