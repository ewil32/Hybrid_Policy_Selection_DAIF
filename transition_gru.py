import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np


class TransitionGRU(keras.Model):

    def __init__(self, latent_dim, action_dim, seq_length, hidden_units, output_dim, batch_size=None, stateful=True, **kwargs):
        super(TransitionGRU, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.seq_length = seq_length
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        self.batch_size = batch_size  # this should be number of policies I think when the agent is being used

        inputs = layers.Input(shape=(None, self.latent_dim + self.action_dim), batch_size=self.batch_size)
        h_states, final_state = layers.GRU(hidden_units, activation="tanh", return_sequences=True, return_state=True, stateful=stateful, name="gru")(inputs)

        # TODO is this correctly getting the last hidden state or the first???
        z_mean = layers.Dense(latent_dim, name="z_mean")(final_state)  # all batch last time step all dimension
        z_log_sd = layers.Dense(latent_dim, name="z_log_sd")(final_state)
        z_stddev = tf.exp(z_log_sd)

        self.transition_model = keras.Model(inputs, [z_mean, z_stddev, final_state, h_states], name="transition")

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
            z_mean, z_stddev, final_state, h_states = self.transition_model(x, training=True)  # Forward pass

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

