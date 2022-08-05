import numpy as np

def random_observation_sequence(env, length):

    observation, info = env.reset()

    observations = [np.array([observation, 0])]
    actions = []
    rewards = []

    for _ in range(length):

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        actions.append(action)
        observations.append(observation)
        rewards.append(reward)

        # pad the end
        if done:
            break


    return np.array(observations), np.array(actions), np.array(rewards)


def transform_observations(observations, observation_max, observation_min, noise_stddev):
    """
    Transform mountain car observations to be in the range 0 to 1
    :param observations:
    :return:
    """

    # https://www.gymlibrary.ml/environments/classic_control/mountain_car_continuous/
    # the standard max and min values
    # observation_max = np.array([0.6, 0.07])
    # observation_min = np.array([-1.2, -0.07])

    # Need to increase the max and min to allow for random noise to be added
    # observation_max = np.array([1.2, 0.14])
    # observation_min = np.array([-2.4, -0.14])

    observations_scaled = (observations - observation_min)/(observation_max - observation_min)

    # add noise
    observation_noisy = observations_scaled + np.random.normal(loc=0, scale=noise_stddev, size=observations_scaled.shape)

    observations_clipped = np.clip(observation_noisy, 0, 1)

    return observations_clipped





# def random_observation_sequence(env, length, num_samples):
#
#     sequences = []
#     rewards = []
#     actions = []
#
#     for i in range(num_samples):
#
#         observation, info = env.reset()
#
#         this_sequence = [observation]
#         this_rewards = []
#         this_actions = []
#
#         for j in range(length):
#
#             action = env.action_space.sample()
#
#             observation, reward, done, info = env.step(action)
#
#             print(observation, reward, done, info)
#             this_actions.append(actions)
#             this_sequence.append(observation)
#             this_rewards.append(reward)
#
#             # pad the end
#             if done:
#                 for j in range(length - j):
#                     this_actionsactions.append(None)
#                     this_sequencesequence.append(None)
#                     this_rewards.append(None)
#
#         sequences.append(this_sequence)
#         rewards.append(this_rewards)
#         actions.append(this_actions)
#
#     env.close()
#
#     return np.array(sequences), np.array(actions), np.array(rewards)
