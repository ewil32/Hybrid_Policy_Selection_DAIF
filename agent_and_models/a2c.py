import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np


class PolicyGradientNetwork(keras.Model):

    def __init__(self,
                 latent_dim,
                 action_dim,
                 dense_units,
                 action_std_dev=0.05,
                 train_epochs=1,
                 show_training=True,
                 discount_factor=0.99,
                 **kwargs):

        super(PolicyGradientNetwork, self).__init__(**kwargs)

        habit_action_inputs = layers.Input(latent_dim)
        h = habit_action_inputs
        for d in dense_units:
            h = layers.Dense(d, activation="relu")(h)

        a_mean = layers.Dense(action_dim, activation="tanh", name="z_mean")(h)
        self.habit_action_model = keras.Model(habit_action_inputs, a_mean, name="habit_action")

        # add the loss over all time steps
        self.loss_tracker = keras.metrics.Sum(name="loss")

        self.action_std_dev = action_std_dev
        self.discount_factor = discount_factor

        # train parameters
        self.train_epochs = train_epochs
        self.show_training = show_training


    def select_action(self, state):
        return self.habit_action_model(state)


    def call(self, inputs, training=None, mask=None):
        return self.habit_action_model(inputs)


    def train(self, pre_obs, actions, cum_rewards, post_obs):

        self.fit(pre_obs,
                 (actions, cum_rewards),
                 epochs=self.train_epochs,
                 verbose=self.show_training,
                 batch_size=pre_obs.shape[0])


    @property
    def metrics(self):
        return [self.loss_tracker]


    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        latent_states, outcomes = data
        true_actions, cum_discounted_reward = outcomes

        with tf.GradientTape() as tape:
            a_mean = self.habit_action_model(latent_states, training=True)  # Forward pass

            log_loss = log_likelihood_gaussian(a_mean, true_actions, self.action_std_dev**2, use_consts=False)
            weighted_log_loss = log_loss * cum_discounted_reward

            # need to multiply by negative one because neural net does gradient descent not ascent
            neg_weighted_log_loss = -1 * weighted_log_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(neg_weighted_log_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(neg_weighted_log_loss)
        return {
            "loss": self.loss_tracker.result()
        }


def compute_discounted_cumulative_reward(rewards, discount_factor):

    gamma = np.ones_like(rewards) * discount_factor
    gamma_t = np.power(gamma, np.arange(rewards.shape[0]).reshape(rewards.shape[0], 1))  # discounted through time

    # discounted rewards starting from the start
    discounted_rewards = np.multiply(rewards, gamma_t)

    n = rewards.shape[0]

    # upper triangular matrix with row i equal to 1/(discount_factor**i)  indexing from 0
    discount_factor_matrix = (np.tril(np.ones((n,n))) * (1/gamma_t)).T

    # cumulative discounted_rewards
    cumulative_discounted_rewards = np.matmul(discount_factor_matrix, discounted_rewards)
    return cumulative_discounted_rewards


def log_likelihood_gaussian(pred, target, variance, use_consts=True):
    #  loss function for log likelihood between predicted and target

    log_prob = -1 * ((pred - target)**2/(2*variance))

    if use_consts:
        const = 0.5*np.log(2*np.pi*variance)
        log_prob += const

    return tf.reduce_sum(log_prob, axis=1)


class A2CAgent:

    def __init__(self, policy_net, value_net, agent_time_ratio):
        self.policy_net = policy_net
        self.value_net = value_net

        self.observation_sequence = []
        self.action_sequence = []
        self.reward_this_run = []

        self.previous_action = None

        self.timestep = 0
        self.agent_time_ratio = agent_time_ratio

    def perceive_and_act(self, observation, reward=None, done=False):

        if done:

            self.reward_this_run.append(reward)
            self.observation_sequence.append(observation)
            self.observation_sequence = np.vstack(self.observation_sequence)
            self.action_sequence = np.array(self.action_sequence)
            self.reward_this_run = np.array(self.reward_this_run).reshape(len(self.reward_this_run), 1)

            pre_obs = self.observation_sequence[0:-1]
            post_obs = self.observation_sequence[1:]

            # train value model
            self.value_net.train(post_obs, self.reward_this_run)

            # calculate advantage
            v_state = self.value_net(pre_obs)
            v_plus_one_state = self.value_net(post_obs)
            advantage = self.reward_this_run + self.value_net.discount_factor * v_plus_one_state - v_state

            # train habit net
            self.policy_net.train(pre_obs, self.action_sequence, advantage, post_obs)

        elif self.timestep % self.agent_time_ratio == 0:

            # select the next action
            action = self.policy_net(observation)
            action = tfp.distributions.MultivariateNormalDiag(loc=action, scale_diag=[self.policy_net.action_std_dev]).sample()
            action = [tf.squeeze(action).numpy()]

            # update previous values
            self.previous_action = action
            self.observation_sequence.append(observation)
            self.action_sequence.append(action)
            self.timestep += 1

            if reward is not None:
                self.reward_this_run.append(reward)

            return action

        else:
            self.timestep += 1
            return self.previous_action

    def reset_all_states(self):

        self.observation_sequence = []
        self.action_sequence = []
        self.reward_this_run = []

        self.timestep = 0
        self.previous_action = None
