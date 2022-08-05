import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers


def identity_encoder(inputs):

    return [inputs, np.ones_like(inputs), inputs]



def identity_decoder(inputs):
    return inputs


class IdentityVAE(keras.Model):
    """
    Implements the identity mapping with standard deviation as all 1s
    """
    def __init__(self, encoder, decoder, reg_mean, reg_stddev, llik_scaling=1, kl_scaling=1, **kwargs):
        super(IdentityVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.reg_mean = reg_mean
        self.reg_stddev = reg_stddev

        self.llik_scaling = llik_scaling
        self.kl_scaling = kl_scaling

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=None, mask=None):
        return inputs

    def train_step(self, data):
        return {
            0
        }

