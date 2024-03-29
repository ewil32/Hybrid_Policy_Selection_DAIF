import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np


class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_stddev = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + z_stddev * epsilon


def create_encoder(input_dim, latent_dim, hidden_units=[16, 8]):

    encoder_inputs = keras.Input(shape=input_dim)

    x = encoder_inputs
    for n in hidden_units:
        x = layers.Dense(n, activation="silu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_std = layers.Dense(latent_dim, name="z_stddev")(x)  # output log of sd
    z_stddev = tf.exp(z_log_std)  # exponentiate to get sd
    z = Sampling()([z_mean, z_stddev])
    encoder = keras.Model(encoder_inputs, [z_mean, z_stddev, z], name="encoder")

    return encoder


def create_decoder(latent_dim, output_dim, hidden_units=[16, 8]):

    latent_inputs = keras.Input(shape=(latent_dim,))

    x = latent_inputs
    for n in hidden_units:
        x = layers.Dense(n, activation="silu")(x)

    decoder_outputs = layers.Dense(output_dim)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


class VAE(keras.Model):
    def __init__(self,
                 encoder,
                 decoder,
                 latent_dim,
                 reg_mean,
                 reg_stddev,
                 recon_stddev=0.05,
                 llik_scaling=1,
                 train_epochs=1,
                 show_training=True,
                 **kwargs):

        super(VAE, self).__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.latent_dim = latent_dim

        self.reg_mean = reg_mean
        self.reg_stddev = reg_stddev

        self.reconstruction_stddev = recon_stddev

        self.llik_scaling = llik_scaling

        self.train_epochs = train_epochs
        self.show_training = show_training

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def compute_loss(self, x=None):
        z_mean, z_stddev, z = self.encoder(x)
        reconstruction = self.decoder(z)

        reconstruction_loss = nll_gaussian(reconstruction, x, self.reconstruction_stddev**2, use_consts=False) * self.llik_scaling

        posterior_dist = tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=z_stddev)
        reg_dist = tfp.distributions.MultivariateNormalDiag(loc=self.reg_mean, scale_diag=self.reg_stddev)
        kl_loss = tfp.distributions.kl_divergence(posterior_dist, reg_dist)

        total_loss = reconstruction_loss + kl_loss
        return total_loss

    def train_step(self, data):

        # unpack data
        x = data
        with tf.GradientTape() as tape:
            z_mean, z_stddev, z = self.encoder(x)
            reconstruction = self.decoder(z)

            reconstruction_loss = nll_gaussian(reconstruction, x, self.reconstruction_stddev**2, use_consts=False) * self.llik_scaling

            posterior_dist = tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=z_stddev)
            reg_dist = tfp.distributions.MultivariateNormalDiag(loc=self.reg_mean, scale_diag=self.reg_stddev)
            kl_loss = tfp.distributions.kl_divergence(posterior_dist, reg_dist)

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),  # TODO should this be total_loss not loss
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def nll_gaussian(pred, target, variance, use_consts=True):

    neg_log_prob = ((pred - target)**2/(2*variance))

    if use_consts:
        const = 0.5*np.log(2*np.pi*variance)
        neg_log_prob += const

    return tf.reduce_sum(neg_log_prob, axis=1)
