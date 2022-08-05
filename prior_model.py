import tensorflow as tf
import keras
from keras import layers
import numpy as np


class PriorModelBellman(keras.Model):


    def __init__(self, observation_dim, learning_rate=0.001, iterate_train=1, discount_factor=0.99):
        super(PriorModelBellman, self).__init__()
        self.observation_dim = observation_dim
        self.learning_rate = learning_rate
        self.iterate_train = iterate_train
        self.discount_factor = discount_factor
        self.train_epochs = 1

        # make the model
        transition_inputs = layers.Input(observation_dim)
        h = layers.Dense(observation_dim * 20, activation="silu")(transition_inputs)
        h = layers.Dense(observation_dim, activation="tanh")(h)

        self.prior_model = keras.Model(transition_inputs, h, name="prior_model")
        self.prior_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())

        self.observations = []
        self.rewards = []


    def extrinsic_kl(self, y):
        return 1.0 - self.forward(y) # map from [-1, 1] to [2, 0]


    def train(self, observations, rewards):

        T = len(observations)

        rewards_stacked = np.stack([rewards[0]]*self.observation_dim).T

        for i in range(self.iterate_train):

            # reducing discount factors through time
            discount_factors = np.power([self.discount_factor]*T, np.arange(T)+1).reshape(observations.shape[0], 1)

            predicted_utility = self.prior_model(observations)

            pred_next_v = tf.concat([predicted_utility[1:], tf.zeros((1, predicted_utility.shape[1]), dtype=predicted_utility.dtype)], axis=0)

            expected_utility = rewards_stacked + discount_factors * pred_next_v

            self.prior_model.fit(observations, expected_utility, epochs=self.train_epochs)
