import numpy as np

def random_observation_sequence(env, length, epsilon=0.5, render_env=False):

    observation = env.reset()

    observations = [observation]
    actions = []
    rewards = []

    action = env.action_space.sample()

    for _ in range(length):

        # change action with epsilon change action, else repeat the same action
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()

        if render_env:
            env.render()

        observation, reward, done, *rest = env.step(action)

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

    observations_scaled = 2*observations_scaled - 1

    # add noise
    observation_noisy = observations_scaled + np.random.normal(loc=0, scale=noise_stddev, size=observations_scaled.shape)

    observations_clipped = np.clip(observation_noisy, -1, 1)

    return observations_clipped