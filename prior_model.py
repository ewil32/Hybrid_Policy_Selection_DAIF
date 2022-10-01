import tensorflow as tf
import keras
from keras import layers
import numpy as np


class PriorModelBellman(keras.Model):

    def __init__(self,
                 observation_dim,
                 output_dim=1,
                 hidden_units=[20],
                 iterate_train=1,
                 discount_factor=0.99,
                 training_epochs=1,
                 show_training=True,
                 use_tanh_on_output=True):

        super(PriorModelBellman, self).__init__()
        self.observation_dim = observation_dim
        self.iterate_train = iterate_train
        self.discount_factor = discount_factor
        self.train_epochs = 1

        self.observations = []
        self.rewards = []

        self.train_epochs = training_epochs
        self.show_training = show_training

        # make the model
        transition_inputs = layers.Input(observation_dim)
        h = transition_inputs
        for n in hidden_units:
            h = layers.Dense(observation_dim * n, activation="silu")(h)

        if use_tanh_on_output:
            h = layers.Dense(output_dim, activation="tanh")(h)
        else:
            h = layers.Dense(output_dim)(h)

        self.prior_model = keras.Model(transition_inputs, h, name="prior_model")
        self.prior_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())


    def call(self, observations):
        return self.prior_model(observations)


    def train(self, observations, rewards):
        """
        :param observations: o_0, o_1, ... , o_n
        :param rewards: list with r_0, r_1, ... , r_n
        :return:
        """

        num_observations = len(observations)

        # print(rewards)

        for i in range(self.iterate_train):

            # TODO Still seems a little strange that we add 0 to the end and discount the way we do but I think it makes sense. Check what predicted utilities are in practice
            utility_t = self.prior_model(observations)
            # utility_t_plus_one = tf.concat([utility_t[1:], tf.zeros((1, self.output_dim), dtype=utility_t.dtype)], axis=0)
            utility_t_plus_one = tf.concat([utility_t[1:], np.zeros((1,1))], axis=0)

            # just have constant gamma
            discount_factors = np.ones_like(utility_t_plus_one) * self.discount_factor


            # OR reducing discount factors through time
            # discount_factors = np.power([self.discount_factor]*num_observations, np.arange(num_observations)).reshape(observations.shape[0], 1)
            # discount_factors = np.flip(discount_factors)

            # print(discount_factors)

            # print(predicted_utility, pred_next_v)

            expected_utility = rewards + discount_factors * utility_t_plus_one

            # print(rewards_stacked)
            # print(discount_factors * utility_t_plus_one)

            self.prior_model.fit(observations, expected_utility, epochs=self.train_epochs, verbose=self.show_training)
