import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


class TransitionModel(keras.Model):

    def __init__(self, latent_dim, action_dim, **kwargs):
        super(TransitionModel, self).__init__(**kwargs)

        transition_inputs = layers.Input(latent_dim + action_dim)
        h = layers.Dense(16)(transition_inputs)
        z_mean = layers.Dense(latent_dim, name="z_mean")(h)
        z_log_sd = layers.Dense(latent_dim, name="z_log_var")(h)
        z_stddev = tf.exp(z_log_sd)

        self.transition_model = keras.Model(transition_inputs, [z_mean, z_stddev], name="transition")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        return self.transition_model(inputs)

    @property
    def metrics(self):
        return [self.kl_loss_tracker]

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        mu, stddev = y

        with tf.GradientTape() as tape:
            z_mean, z_stddev = self.transition_model(x, training=True)  # Forward pass

            # Compute the loss value
            pred_dist = tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=z_stddev)
            true_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=stddev)

            # TODO make sure this is the correct order of terms
            kl_loss = tfp.distributions.kl_divergence(pred_dist, true_dist)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(kl_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "kl_loss": self.kl_loss_tracker.result()
        }
