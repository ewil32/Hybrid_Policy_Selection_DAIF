import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np


class Sampling(layers.Layer):
    """Uses (z_mean, z_stddev) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_stddev = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + z_stddev * epsilon


def create_conv_encoder(input_dim, latent_dim, conv_shapes=[3, 3, 3], num_filters=[32, 64, 64], dense_units=[16]):

    encoder_inputs = keras.Input(shape=input_dim)

    x = encoder_inputs
    for n in range(len(conv_shapes)):
        filter_shape = conv_shapes[n]
        num = num_filters[n]
        x = layers.Conv2D(num, filter_shape, activation="relu", strides=2, padding="same")(x)

    x = layers.Flatten()(x)

    for d in dense_units:
        x = layers.Dense(d, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_std = layers.Dense(latent_dim, name="z_stddev")(x)  # output log of sd
    z_stddev = tf.exp(z_log_std)  # exponentiate to get sd
    z = Sampling()([z_mean, z_stddev])
    encoder = keras.Model(encoder_inputs, [z_mean, z_stddev, z], name="encoder")

    return encoder


def create_conv_decoder(latent_dim, output_shape, deconv_shapes=[3, 3, 3], num_filters=[64, 64, 32], dense_units=[16], rgb=True):

    latent_inputs = keras.Input(shape=(latent_dim,))

    x = latent_inputs

    for d in dense_units:
        x = layers.Dense(d, activation="relu")(x)

    first_shape = (deconv_shapes[0], deconv_shapes[0], num_filters[0])
    x = layers.Reshape(first_shape)(x)

    for n in range(len(deconv_shapes)):
        filter_shape = deconv_shapes[n]
        num = num_filters[n]
        x = layers.Conv2DTranspose(num, filter_shape, activation="relu", strides=2, padding="same")(x)

    if rgb:
        decoder_outputs = x
    else:
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


class ConvVAE(keras.Model):
    def __init__(self, encoder, decoder, latent_dim, reg_mean, reg_stddev, recon_stddev=0.05, llik_scaling=1, kl_scaling=1, train_epochs=1, show_training=False, **kwargs):

        super(ConvVAE, self).__init__(**kwargs)
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
        self.kl_scaling = kl_scaling

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


    def train_step(self, data):

        # unpack data
        # x, reg_vals = data
        x = data
        # reg_mean, reg_stddev = reg_vals
        with tf.GradientTape() as tape:
            z_mean, z_stddev, z = self.encoder(x)
            reconstruction = self.decoder(z)

            # TODO Fix this to be a real loss function
            reconstruction_loss = nll_gaussian(reconstruction, x, self.reconstruction_stddev, use_consts=False)

            posterior_dist = tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=z_stddev)
            reg_dist = tfp.distributions.MultivariateNormalDiag(loc=self.reg_mean, scale_diag=self.reg_stddev)
            kl_loss = tfp.distributions.kl_divergence(posterior_dist, reg_dist) * self.kl_scaling

            # kl_loss = tf.reduce_sum(kl_loss, axis=1)
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

    return tf.reduce_sum(neg_log_prob)
