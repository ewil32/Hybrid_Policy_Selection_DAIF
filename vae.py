import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


class Sampling(layers.Layer):
    """Uses (z_mean, z_stddev) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_stddev = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + z_stddev * epsilon


def create_encoder(latent_dim, input_dim, hidden_units=[16, 8]):

    encoder_inputs = keras.Input(shape=input_dim)

    x = encoder_inputs
    for n in hidden_units:
        x = layers.Dense(n, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_std = layers.Dense(latent_dim, name="z_stddev")(x)  # output log of sd
    z_stddev = tf.exp(z_log_std)  # exponentiate to get sd
    z = Sampling()([z_mean, z_stddev])
    encoder = keras.Model(encoder_inputs, [z_mean, z_stddev, z], name="encoder")

    return encoder


def create_decoder(latent_dim, input_dim, hidden_units=[16, 8]):

    latent_inputs = keras.Input(shape=(latent_dim,))

    x = latent_inputs
    for n in hidden_units:
        x = layers.Dense(n, activation="relu")(x)

    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_stddev, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstruction), axis=0)
            reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
            # reconstruction_loss = reconstruction_loss * data.shape[1]  # Think I needed this to balance out for the data set. It makes it work but I don't know why

            # kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            kl_loss = -tf.math.log(z_stddev) + 0.5 * (tf.square(z_stddev) + tf.square(z_mean) - 1)

            # kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = tf.reduce_sum(kl_loss, axis=1)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
