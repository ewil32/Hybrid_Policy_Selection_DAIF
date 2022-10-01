
import matplotlib.pyplot as plt
import tensorflow as tf

from train_agent import train_single_agent_car_racing

tf.config.run_functions_eagerly(True)

import gym
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pandas as pd

from conv_vae import ConvVAE, create_conv_encoder, create_conv_decoder
from transition_gru import TransitionGRU
from recurrent_agent import DAIFAgentRecurrent
from prior_model import PriorModelBellman
from ddpg import *



latent_dim = 32
pln_hrzn = 2
action_dim = 3

e = create_conv_encoder(input_dim=(64, 64, 3), latent_dim=latent_dim, num_filters=[16, 32, 32], dense_units=[32])
d = create_conv_decoder(latent_dim=latent_dim, output_shape=(64, 64, 3), deconv_shapes=[8, 16, 32], num_filters=[16, 16, 3], dense_units=[8 * 8 * 16])

cvae = ConvVAE(e, d, latent_dim=latent_dim, reg_mean=[0] * latent_dim, reg_stddev=[1] * latent_dim, show_training=True)
cvae.compile(optimizer=tf.keras.optimizers.Adam())

tran = TransitionGRU(latent_dim, action_dim=action_dim, hidden_units=2*latent_dim*pln_hrzn, output_dim=latent_dim)
tran.compile(optimizer=tf.keras.optimizers.Adam())

# unscaled prior mean and prior stddev
prior_model = PriorModelBellman(latent_dim)

daifa = DAIFAgentRecurrent(prior_model,
                           cvae,
                           tran,
                           None,
                           agent_time_ratio=50,
                           planning_horizon=pln_hrzn,
                           use_kl_extrinsic=True,
                           use_kl_intrinsic=True,
                           use_FEEF=False,
                           train_habit_net=False,
                           train_prior_model=True,
                           train_tran=True,
                           train_during_episode=True,
                           train_with_replay=True,
                           use_fast_thinking=False,
                           n_policies=100,
                           n_policy_candidates=10)


# train the agent on the env
env = gym.make("CarRacing-v2", new_step_api=True)
agent, results = train_single_agent_car_racing(env, daifa, num_episodes=10, render_env=True)
#%%
