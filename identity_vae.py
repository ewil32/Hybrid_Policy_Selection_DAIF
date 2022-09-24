import keras
import tensorflow as tf
import numpy as np

def identity_encoder(inputs):

    return [tf.convert_to_tensor(inputs), tf.ones_like(inputs), tf.convert_to_tensor(inputs)]


def identity_decoder(inputs):
    return tf.convert_to_tensor(inputs)


class IdentityVAE(keras.Model):
    """
    Implements the identity mapping with standard deviation as all 1s
    """
    def __init__(self,
                 encoder,
                 decoder,
                 latent_dim,
                 reg_mean,
                 reg_stddev,
                 recon_stddev=0.05,
                 llik_scaling=1,
                 train_epochs=1,
                 show_training=True
                 **kwargs):

        super(IdentityVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        self.latent_dim = latent_dim

        self.reg_mean = reg_mean
        self.reg_stddev = reg_stddev
        self.recon_stddev = recon_stddev

        self.llik_scaling = llik_scaling
        self.kl_scaling = kl_scaling

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
        ]

    def call(self, inputs, training=None, mask=None):
        return inputs

    def train_step(self, data):
        return {
            "total_loss": 0
        }