import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


class BasicDDPG:
    """
    Most of the code in this class is taken and adapted from https://keras.io/examples/rl/ddpg_pendulum/
    """

    def __init__(self, actor, critic, target_actor, target_critic, tau,
                 buffer_capacity=100000,
                 batch_size=64,
                 discount_factor=0.99,
                 observation_dim=2,
                 action_dim=1,
                 critic_optimizer="adam",
                 actor_optimizer="adam",
                 show_training=False):

        self.actor_model = actor
        self.critic_model = critic

        self.target_actor = target_actor
        self.target_critic = target_critic

        # self.buffer = buffer
        self.tau = tau
        self.show_training = show_training

        # BUFFER

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, observation_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, observation_dim))

        self.discount_factor = discount_factor

        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

    def select_action(self, state):
        return self.actor_model(state)

    def update_actor_target(self):
        update_target(self.target_actor.variables, self.actor_model.variables, self.tau)

    def update_critic_target(self):
        update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def record_from_lists(self, obs, actions, rewards, next_obs):
        for n in range(len(obs)):
            self.record((obs[n], actions[n], rewards[n], next_obs[n]))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # clears the buffer
    def clear(self):
        self.state_buffer= []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []

    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.discount_factor * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def train(self, pre_obs, actions, rewards, post_obs):

        self.record_from_lists(pre_obs, actions, rewards, post_obs)

        self.learn()
        self.update_actor_target()
        self.update_critic_target()

        if self.show_training:
            print(np.min(self.actor_model(self.state_buffer)))
            print(np.max(self.actor_model(self.state_buffer)))
            print()


def get_actor(observation_dim, action_max, hidden_units=[16, 32, 16]):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(observation_dim,))
    out = inputs
    for h in hidden_units:
        out = layers.Dense(h, activation="relu")(out)

    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * action_max
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(observation_dim, action_dim, state_hidden_units=[16, 32], action_hidden_units=[32], out_hidden_units=[128, 128]):
    # State as input
    state_input = layers.Input(shape=observation_dim)
    state_out = state_input
    for h in state_hidden_units:
        state_out = layers.Dense(h, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=action_dim)
    action_out = action_input
    for h in action_hidden_units:
        action_out = layers.Dense(h, activation="relu")(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    # was 256
    out = concat
    for h in out_hidden_units:
        out = layers.Dense(h, activation="relu")(out)

    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DDPGAgent(BasicDDPG):

    def __init__(self, ou_noise, agent_time_ratio, **kwargs):
        super(DDPGAgent, self).__init__(**kwargs)

        self.ou_noise = ou_noise

        self.previous_action = None
        self.previous_observation = None
        self.reward_this_run = []

        self.timestep = 0
        self.agent_time_ratio = agent_time_ratio

    def perceive_and_act(self, observation, reward=None, done=False):

        if done:
            self.record((self.previous_observation, self.previous_action, reward, observation))

            self.learn()
            self.update_actor_target()
            self.update_critic_target()

            self.reward_this_run.append(reward)

        elif self.timestep % self.agent_time_ratio == 0:
            # select the next action
            action = tf.squeeze(self.select_action(observation)).numpy()
            action += self.ou_noise()

            # train if we need to
            if self.previous_action is not None:
                self.record((self.previous_observation, self.previous_action, reward, observation))

                self.learn()
                self.update_actor_target()
                self.update_critic_target()

                self.reward_this_run.append(reward)

            # update previous values
            self.previous_action = action
            self.previous_observation = observation
            self.timestep += 1

            return action

        else:
            self.timestep += 1
            return self.previous_action


    def reset_all_states(self, reset_buffer=False):

        self.ou_noise.reset()
        self.previous_observation = None
        self.previous_action = None
        self.timestep = 0
        self.reward_this_run = []

        if reset_buffer:
            self.buffer.clear()

