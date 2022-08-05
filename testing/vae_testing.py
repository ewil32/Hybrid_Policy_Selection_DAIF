
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import gym

from vae_recurrent import VAE, create_decoder, create_encoder
from util import random_observation_sequence, transform_observations

env = gym.make('MountainCarContinuous-v0')
env.action_space.seed(42)

observation_max = np.array([0.6, 0.07])
observation_min = np.array([-1.2, -0.07])

o, a, r = random_observation_sequence(env, 10000)
o_scaled = transform_observations(o, observation_max, observation_min, [0, 0])
o_scaled

enc = create_encoder(2, 2, [20])
dec = create_decoder(2, 2, [20])

vae = VAE(enc, dec, [0, 0], [0.3, 0.3])
vae.compile(optimizer=tf.keras.optimizers.Adam())

vae.fit(o_scaled, epochs=20)

(o_scaled)