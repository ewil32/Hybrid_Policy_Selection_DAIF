import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np


class HabitualAction(keras.Model):

    def __init__(self,
                 latent_dim,
                 action_dim,
                 dense_units,
                 action_std_dev=0.05,
                 train_epochs=1,
                 show_training=True,
                 discount_factor=0.99,
                 **kwargs):

        super(HabitualAction, self).__init__(**kwargs)

        habit_action_inputs = layers.Input(latent_dim)
        h = habit_action_inputs
        for d in dense_units:
            h = layers.Dense(d, activation="relu")(h)

        a_mean = layers.Dense(action_dim, activation="tanh", name="z_mean")(h)
        # a_log_sd = layers.Dense(action_dim, name="z_log_var")(h)
        # a_stddev = tf.exp(a_log_sd)

        # self.habit_action_model = keras.Model(habit_action_inputs, [a_mean, a_stddev], name="habit_action")
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

        # TODO what do I assume the
        with tf.GradientTape() as tape:
            # a_mean, a_stddev = self.habit_action_model(latent_states, training=True)  # Forward pass
            a_mean = self.habit_action_model(latent_states, training=True)  # Forward pass

            # print(a_mean, true_actions)

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
    # print(gamma.shape)
    gamma_t = np.power(gamma, np.arange(rewards.shape[0]).reshape(rewards.shape[0], 1))  # discounted through time

    # print(gamma_t)
    # print(gamma_t.shape)

    # discounted rewards starting from the start
    discounted_rewards = np.multiply(rewards, gamma_t)

    # print(rewards)
    # print(discounted_rewards)

    n = rewards.shape[0]

    # upper trianglur matrix with row i equal to 1/(discount_factor**i)  indexing from 0
    discount_factor_matrix = (np.tril(np.ones((n,n))) * (1/gamma_t)).T
    discount_factor_matrix

    # cumulative discounted_rewards
    cumulative_discounted_rewards = np.matmul(discount_factor_matrix, discounted_rewards)
    return cumulative_discounted_rewards



def log_likelihood_gaussian(pred, target, variance, use_consts=True):

    log_prob = -1 * ((pred - target)**2/(2*variance))

    if use_consts:
        const = 0.5*np.log(2*np.pi*variance)
        log_prob += const

    return tf.reduce_sum(log_prob, axis=1)