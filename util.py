import numpy as np
import pandas as pd

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


def transform_observations(observations, observation_max=None, observation_min=None, noise_stddev=None):
    """
    https://www.gymlibrary.ml/environments/classic_control/mountain_car_continuous/

    Transform mountain car observations to be in the range 0 to 1
    :param observations:
    :return:
    """

    if observation_max is not None and observation_min is not None:
        observations_scaled = (observations - observation_min)/(observation_max - observation_min)
        observations_scaled = 2*observations_scaled - 1
    else:
        observations_scaled = observations

    # add noise
    if noise_stddev is not None:
        observation_noisy = observations_scaled + np.random.normal(loc=0, scale=noise_stddev, size=observations_scaled.shape)
        observations_clipped = np.clip(observation_noisy, -1, 1)
    else:
        observations_clipped = observations_scaled

    return observations_clipped


def transform_image(img, x_min, x_max, y_min, y_max):
    img_out = img/255

    img_out = img_out[x_min:x_max, y_min:y_max, :]

    return img_out


def test_policy(env, policy_func, observation_max, observation_min, obs_stddev, num_episodes, num_action_repeats, show_env=False):

    all_rewards = []
    all_times = []
    all_num_actions = []

    rows = []

    for i in range(num_episodes):

        obs = env.reset()

        if show_env:
            env.render()

        done = False
        rewards = []
        t = 0

        while not done:

            obs = obs.reshape(1, obs.shape[0])
            obs = transform_observations(obs, observation_max, observation_min, obs_stddev)

            action = policy_func(obs)
            action = action.numpy()
            # print(action)

            for k in range(num_action_repeats):
                obs, reward, done, info = env.step(action)

                t += 1

                if show_env:
                    env.render()

            rewards.append(reward)

        rows.append([np.sum(rewards), t, t//num_action_repeats])
        # all_rewards.append(np.sum(rewards))
        # all_times.append(t)
        # all_num_actions.append(t//num_action_repeats)

    env.close()

    results = pd.DataFrame(rows, columns=["reward", "timesteps", "num_actions"])

    return results


def habit_policy(agent):

    def f(obs):
        action = agent.select_habit_policy(obs)
        return action

    return f


