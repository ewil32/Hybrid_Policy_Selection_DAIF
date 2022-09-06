import tensorflow as tf
import keras
from keras import layers
import numpy as np


class PriorModelBellman(keras.Model):


    def __init__(self,
                 observation_dim,
                 iterate_train=1,
                 discount_factor=0.99,
                 training_epochs=1,
                 show_training=True):

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
        h = layers.Dense(observation_dim * 20, activation="silu")(transition_inputs)
        h = layers.Dense(observation_dim, activation="tanh")(h)

        self.prior_model = keras.Model(transition_inputs, h, name="prior_model")
        self.prior_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())

    def call(self, observations):
        return self.prior_model(observations)

    def extrinsic_kl(self, observations):
        return 1.0 - self(observations)  # map from [-1, 1] to [2, 0]

    def train(self, observations, rewards):
        """

        :param observations: o_0, o_1, ... , o_n
        :param rewards: list with r_0, r_1, ... , r_n
        :return:
        """

        num_observations = len(observations)

        # expand rewards to have the same dimension as observation dimension and transpose to give [num_observations, observation_dimension
        rewards_stacked = np.stack([rewards]*self.observation_dim).T

        for i in range(self.iterate_train):

            # reducing discount factors through time
            discount_factors = np.power([self.discount_factor]*num_observations, np.arange(num_observations)).reshape(observations.shape[0], 1)
            discount_factors = np.flip(discount_factors)

            # print(discount_factors)

            # TODO Still seems a little strange that we add 0 to the end and discount the way we do but I think it makes sense. Check what predicted utilities are in practice
            utility_t = self.prior_model(observations)
            utility_t_plus_one = tf.concat([utility_t[1:], tf.zeros((1, self.observation_dim), dtype=utility_t.dtype)], axis=0)

            # print(predicted_utility, pred_next_v)

            expected_utility = rewards_stacked + discount_factors * utility_t_plus_one

            # print(rewards_stacked)
            # print(discount_factors * utility_t_plus_one)

            self.prior_model.fit(observations, expected_utility, epochs=self.train_epochs, verbose=self.show_training)
