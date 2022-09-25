import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np


class TransitionFeedForward(keras.Model):

    def __init__(self,
                 latent_dim,
                 action_dim,
                 hidden_units,
                 output_dim,
                 train_epochs=1,
                 show_training=True,
                 **kwargs):

        super(TransitionFeedForward, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        # build the network
        inputs = layers.Input(shape=(None, self.latent_dim + self.action_dim))
        h = inputs
        for units in hidden_units:
            h = layers.Dense(units, activation="relu")(h)

        # TODO is this correctly getting the last hidden state or the first???
        z_mean = layers.Dense(latent_dim, name="z_mean")(h)  # all batch last time step all dimension
        z_log_sd = layers.Dense(latent_dim, name="z_log_sd")(h)
        z_stddev = tf.exp(z_log_sd)

        self.transition_model = keras.Model(inputs, [z_mean, z_stddev], name="transition")

        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        # important attributes
        self.train_epochs = train_epochs
        self.show_training = show_training


    def call(self, inputs, training=None, mask=None):

        # extract the initial state and hidden state
        return self.transition_model(inputs)

    def compute_loss(self, inputs, targets):

        x = inputs
        mu, stddev = targets

        z_mean, z_stddev = self.transition_model(x)  # Forward pass

        # Compute the loss value
        pred_dist = tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=z_stddev)
        true_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=stddev)

        # TODO make sure this is the correct order of terms
        kl_loss = tfp.distributions.kl_divergence(pred_dist, true_dist)
        return kl_loss


    @property
    def metrics(self):
        return [self.kl_loss_tracker]

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        obs, targets = data
        mu, stddev = targets

        with tf.GradientTape() as tape:
            z_mean, z_stddev = self.transition_model(obs, training=True)  # Forward pass

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
