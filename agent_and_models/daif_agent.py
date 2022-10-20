import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class DAIFAgent:

    def __init__(self,
                 prior_model,
                 vae,
                 tran,
                 habitual_action_net,
                 given_prior_mean=None,
                 given_prior_stddev=None,
                 agent_time_ratio=6,
                 actions_to_execute_when_exploring=2,
                 planning_horizon=15,
                 n_policies=1500,
                 n_cem_policy_iterations=2,
                 n_policy_candidates=70,
                 train_vae=True,
                 train_tran=True,
                 train_prior_model=True,
                 train_habit_net=True,
                 train_with_replay=True,
                 train_during_episode=True,
                 use_efe_extrinsic=True,
                 use_kl_intrinsic=True,
                 use_FEEF=True,
                 use_habit_policy=False,
                 uncertainty_tolerance=0.05,
                 habit_model_type="name_of_model",
                 min_rewards_needed_to_train_prior=0,
                 prior_model_scaling_factor=1):
        """
        Initialiser for the agent.
        :param prior_model:
        :param vae:
        :param tran:
        :param habitual_action_net:
        :param given_prior_mean:
        :param given_prior_stddev:
        :param agent_time_ratio:
        :param actions_to_execute_when_exploring:
        :param planning_horizon:
        :param n_policies:
        :param n_cem_policy_iterations:
        :param n_policy_candidates:
        :param train_vae:
        :param train_tran:
        :param train_prior_model:
        :param train_habit_net:
        :param train_with_replay:
        :param train_during_episode:
        :param use_efe_extrinsic:
        :param use_kl_intrinsic:
        :param use_FEEF:
        :param use_habit_policy:
        :param uncertainty_tolerance:
        :param habit_model_type:
        :param min_rewards_needed_to_train_prior:
        :param prior_model_scaling_factor:
        """

        super(DAIFAgent, self).__init__()

        # parameters for slow policy planning
        self.planning_horizon = planning_horizon
        self.n_policy_candidates = n_policy_candidates
        self.n_policies = n_policies
        self.n_cem_policy_iterations = n_cem_policy_iterations

        # flags for whether or not we are training models or using pretrained models and when we should train
        self.train_vae = train_vae
        self.train_tran = train_tran
        self.train_habit_net = train_habit_net
        self.train_prior = train_prior_model
        self.train_with_replay = train_with_replay
        self.train_during_episode = train_during_episode

        # do we use the kl divergence for extrinsic vs intrinsic
        self.use_kl_intrinsic = use_kl_intrinsic
        self.use_efe_extrinsic = use_efe_extrinsic

        # do we use the FEEF or EFE?
        self.use_FEEF = use_FEEF

        # given prior values
        self.given_prior_mean = given_prior_mean
        self.given_prior_stddev = given_prior_stddev
        self.prior_model_scaling_factor = prior_model_scaling_factor

        # full vae
        self.model_vae = vae
        self.tran = tran
        self.prior_model = prior_model
        self.habit_action_model = habitual_action_net

        # how much is the agents planning time compressed compared to the simulation time
        self.agent_time_ratio = agent_time_ratio
        self.actions_to_execute_when_exploring = actions_to_execute_when_exploring
        self.time_step = 0
        self.exploring = False

        # track the hidden state of the transition gru model so we can use it to train
        self.tran_hidden_state = None
        self.tran_hidden_state_pre_exploring = None
        self.prev_tran_hidden_state = None

        # store the full observations for the episode so we can train using replay
        self.full_observation_sequence = []
        self.full_action_sequence = []
        self.full_reward_sequence = []

        # store the observations while the agent is in exploration mode
        self.exploring_observation_sequence = []
        self.exploring_action_sequence = []
        self.exploring_reward_sequence = []

        # store the observations at the world time scale
        self.env_time_scale_observations = []

        self.policy_left_to_execute = [None]
        self.previous_observation = None
        self.action_being_executed = None
        self.action_being_executed = 0

        self.use_habit_policy = use_habit_policy
        self.habit_model_type = habit_model_type
        self.uncertainty_tolerance = uncertainty_tolerance
        self.num_habit_choices = 0
        self.habit_policy_streak = 0

        # Normally 0 but this is just a bad parameter and I don't know if it should exist
        self.min_rewards_needed_to_train_prior = min_rewards_needed_to_train_prior

    def perceive_and_act(self, observation, reward, done):
        """

        :param observation:
        :param reward:
        :param done:
        :return:
        """

        # track the world time scale observation sequence
        self.env_time_scale_observations.append(observation)

        # if the episode is finished, then do any training on the full data set
        if done:

            print("Number of habit choices:", self.num_habit_choices)
            print("Number of actions total:", len(self.full_action_sequence))

            # add the final observation and reward we observed to the sequences
            self.full_observation_sequence.append(observation)
            self.full_reward_sequence.append(reward)

            if self.train_with_replay:
                print("training on full data")
                # Call the training function on the observation sequences to train everything we need to train
                self.train_models(np.vstack(self.full_observation_sequence),
                                  np.vstack(self.full_action_sequence),
                                  np.array(self.full_reward_sequence),
                                  None)

        # Otherwise are we at a point where we can reconsider our policy and maybe train the world model
        elif self.time_step % self.agent_time_ratio == 0:

            # add the observation to the sequence
            self.full_observation_sequence.append(observation)

            # add the reward only if it's not the first observation
            if self.time_step != 0:
                self.full_reward_sequence.append(reward)

            # We only update the model during the episode when we were exploring using the planning method and we have executed all of the actions in the policy
            if self.exploring and len(self.policy_left_to_execute) == 0:

                # now we're no longer exploring
                self.exploring = False

                if self.train_during_episode:

                    # the actions done while exploring were the last self.actions_to_execute_when_exploring
                    self.exploring_action_sequence = self.full_action_sequence[-1*self.actions_to_execute_when_exploring:]
                    self.exploring_reward_sequence = self.full_reward_sequence[-1*self.actions_to_execute_when_exploring:]
                    self.exploring_observation_sequence = self.full_observation_sequence[-1*(self.actions_to_execute_when_exploring + 1):]

                    # Call the training function on the observation sequences to train everything we need to train
                    self.train_models(np.vstack(self.exploring_observation_sequence),
                                      np.vstack(self.exploring_action_sequence),
                                      np.array(self.exploring_reward_sequence),
                                      tran_hidden_state_pre_obs=self.tran_hidden_state_pre_exploring)


            # Predict the expected observation based on the previously executed action
            action_as_array = np.array(self.action_being_executed).reshape(1, self.tran.action_dim)
            expected_observation, self.tran_hidden_state = self.predict_next_observation(self.previous_observation, action_as_array, self.prev_tran_hidden_state)

            # Now we select our action. If we aren't exploring then either we act out of habit or we might need to explore
            # I think I can check this based on whether or not there are actions left to execute in the current policy
            if not self.exploring:

                # first observation will have no previous observation
                if self.use_habit_policy and self.previous_observation is None:
                    self.policy_left_to_execute = self.select_habit_policy(observation)
                    self.policy_left_to_execute = self.policy_left_to_execute.numpy().tolist()  # tf tensor to list

                    self.habit_policy_streak += 1

                # we need to see what the generative model now thinks about what the expected current observation is
                elif self.use_habit_policy and np.allclose(observation, expected_observation, atol=self.uncertainty_tolerance):  # within some tolerance

                    # use habit network to select policy
                    self.policy_left_to_execute = self.select_habit_policy(observation)
                    self.policy_left_to_execute = self.policy_left_to_execute.numpy().tolist()

                    self.num_habit_choices += 1
                    self.habit_policy_streak += 1

                # the generative model is surprised so we should use the CEM for planning out a policy that balances exploration and exploitation
                else:

                    # train the fast thinker because now we're about to explore
                    if self.train_during_episode and self.habit_policy_streak > 0:
                        # the actions done while exploring were the last self.actions_to_execute_when_exploring
                        habit_action_sequence = self.full_action_sequence[-1*self.habit_policy_streak:]
                        habit_reward_sequence = self.full_reward_sequence[-1*self.habit_policy_streak:]
                        habit_observation_sequence = self.full_observation_sequence[-1*(self.habit_policy_streak + 1):]

                        # Call the training function on the observation sequences to train everything we need to train
                        self.train_models(np.vstack(habit_observation_sequence),
                                          np.vstack(habit_action_sequence),
                                          np.array(habit_reward_sequence),
                                          tran_hidden_state_pre_obs=None,
                                          train_vae=False,
                                          train_tran=False)

                    # reset the fast thinking streak
                    self.habit_policy_streak = 0

                    # select a policy with CEM thinking
                    policy_dist = self.select_CEM_policy(observation)
                    policy = policy_dist.sample().numpy()
                    policy = policy.reshape(policy.shape[0], self.tran.action_dim).tolist()
                    self.policy_left_to_execute = policy[0: self.actions_to_execute_when_exploring]

                    # track the hidden state so we can use it once we've finished exploring
                    self.tran_hidden_state_pre_exploring = self.tran_hidden_state
                    self.exploring = True

            # finally update the previous observation and action to be the one we just had/did
            self.previous_observation = observation
            self.prev_tran_hidden_state = self.tran_hidden_state
            self.action_being_executed = self.policy_left_to_execute[0]
            self.full_action_sequence.append(self.action_being_executed)
            self.policy_left_to_execute.pop(0)

        # final updates increment the current timestep and return the action specified by the policy
        self.time_step += 1

        return self.action_being_executed

    def predict_next_observation(self, obs, action, tran_hidden_state):
        """

        :param obs:
        :param action:
        :param tran_hidden_state:
        :return:
        """

        if obs is None:
            return None, None
        else:
            # get the agents internal latent state z
            z_mean, z_std, z = self.model_vae.encoder(obs)
            z_mean = z_mean.numpy()
            z_plus_action = np.concatenate([z_mean, action], axis=1)
            z_plus_action = z_plus_action.reshape((1, 1, z_plus_action.shape[1]))

            # use transition to predict next parameters of next state distribution
            next_latent_mean, next_latent_sd, next_hidden_state, _ = self.tran((z_plus_action, tran_hidden_state))

            # use likelihood to find predicted next observation
            next_observation = self.model_vae.decoder(next_latent_mean)
            return next_observation.numpy(), next_hidden_state

    def reset_tran_hidden_state(self):
        """
        Function to reset the hidden state of the transition model when we want to train on the full data set
        :return: None
        """
        self.tran_hidden_state = None

    def reset_all_states(self):
        """
        Resets all of the agents states. Used when training the agent on a new episode
        :return: None
        """
        self.time_step = 0
        self.exploring = False

        # track the hidden state of the transition gru model so we can use it to train
        self.tran_hidden_state = None

        # store the full observations for the episode so we can train using replay
        self.full_observation_sequence = []
        self.full_action_sequence = []
        self.full_reward_sequence = []

        # store the observations while the agent is in exploration mode
        self.exploring_observation_sequence = []
        self.exploring_action_sequence = []
        self.exploring_reward_sequence = []

        self.policy_left_to_execute = []
        self.previous_observation = None

        self.num_habit_choices = 0
        self.habit_policy_streak = 0


    def train_models(self, observations_full, actions, rewards, tran_hidden_state_pre_obs, train_vae=True, train_tran=True, train_prior=True, train_habit=True):
        """

        :param observations_full:
        :param actions:
        :param rewards:
        :param tran_hidden_state_pre_obs:
        :param train_vae:
        :param train_tran:
        :param train_prior:
        :param train_habit:
        :return:
        """

        # Split observations into pre and post.
        pre_observations = observations_full[:-1]
        post_observations = observations_full[1:]

        # find the actual observed latent states using the vae
        pre_latent_mean, pre_latent_stddev, pre_latent = self.model_vae.encoder(pre_observations)
        post_latent_mean, post_latent_stddev, post_latent = self.model_vae.encoder(post_observations)

        #### TRAIN THE TRANSITION MODEL ####
        if self.train_tran and train_tran:

            num_observations = pre_latent_mean.shape[0]
            action_dim = actions.shape[1]
            latent_dim = self.model_vae.latent_dim

            # set up the input training data that we use to train the transition model
            z_train = np.concatenate([np.array(pre_latent_mean), actions], axis=1)

            # we use the sequence to find the right hidden states to use as input
            z_train_seq = z_train.reshape((1, num_observations, latent_dim + action_dim))
            z_train_singles = z_train.reshape((num_observations, 1, latent_dim + action_dim))

            # the previous hidden state is the memory after observing some sequences but it might be None if we're just starting
            if tran_hidden_state_pre_obs is None:
                tran_hidden_state_pre_obs = np.zeros((1, self.tran.hidden_units))

            # find the hidden states at t=0, t=1, t=2, ..., t=num_observations - 1
            _, _, _, h_states = self.tran((z_train_seq, tran_hidden_state_pre_obs))

            # squeeze so we make the shape [num_observations, hidden_units]
            h_states = tf.squeeze(h_states)

            # exclude the last state as this will become the hidden state later on. next hidden state will become our new memory
            h_states_for_training = h_states[:-1]

            # add the current hidden state we saved to the start. This has h0, h1, h2, .. h=num_observations - 1
            h_states_for_training = tf.concat([tran_hidden_state_pre_obs, h_states_for_training], axis=0)

            # use the hidden states with the pre and post observations to train transition model
            self.tran.fit((z_train_singles, h_states_for_training), (post_latent_mean, post_latent_stddev), epochs=self.tran.train_epochs, verbose=self.tran.show_training, batch_size=z_train_singles.shape[0])

            # now find the new predicted hidden state that we will use for finding the policy
            _, _, final_hidden_state, h_states = self.tran((z_train_seq, tran_hidden_state_pre_obs))

            z_pred, _, _, _ = self.tran((z_train_singles, h_states_for_training))

            # update the transition model hidden states
            self.prev_tran_hidden_state = h_states[:, -2, :]
            self.tran_hidden_state = final_hidden_state

        #### TRAIN THE VAE ####
        if self.train_vae and train_vae:
            self.model_vae.fit(pre_observations, epochs=self.model_vae.train_epochs, verbose=self.model_vae.show_training, batch_size=pre_observations.shape[0])

        #### TRAIN THE PRIOR MODEL ####
        if self.train_prior and train_prior:
            if max(rewards) > self.min_rewards_needed_to_train_prior:
                # self.prior_model.train(post_observations, rewards)
                self.prior_model.train(post_latent_mean, rewards)

        #### TRAIN THE HABIT ACTION NET ####
        if self.train_habit_net and train_habit:

            # training is different depending on the habit model because we need to calculate advantage for A2C
            if self.habit_model_type == "A2C":

                # ADVANTAGE
                v_state = self.prior_model(pre_latent_mean)
                v_plus_one_state = self.prior_model(post_latent_mean)
                advantage = rewards + self.prior_model.discount_factor * v_plus_one_state - v_state

                # DDPG and policy gradient interface with same function
                self.habit_action_model.train(pre_latent_mean, actions, advantage, post_latent_mean)

            if self.habit_model_type == "DDPG":
                self.habit_action_model.train(pre_latent_mean, actions, rewards, post_latent_mean)

    def select_habit_policy(self, observation):
        """

        :param observation:
        :return:
        """

        # get agents latent state and use it find action with habit model
        latent_state,  _, _ = self.model_vae.encoder(observation)
        action = self.habit_action_model.select_action(latent_state)

        # if we're using A2C we should sample from distribution or DDPG is deterministic
        if self.habit_model_type == "A2C":
            return tfp.distributions.MultivariateNormalDiag(loc=action, scale_diag=[self.habit_action_model.action_std_dev]).sample()
        elif self.habit_model_type == "DDPG":
            return action

    def select_CEM_policy(self, observation):
        """
        :param observation: needs to be [n, observation_dim] shape np array or tf tensor
        :return:
        """

        # get the latent state from this observation
        _,  _, latent_state = self.model_vae.encoder(observation)

        # select the policy
        policy_mean, policy_stddev = self.cem_policy_optimisation(latent_state)

        # return a distribution that we can sample from
        return tfp.distributions.MultivariateNormalDiag(loc=policy_mean, scale_diag=policy_stddev)

    def cem_policy_optimisation(self, latent_z):
        """

        :param latent_z:
        :return:
        """

        # need to change these two if the policy dimension changes
        mean_best_policies = tf.zeros((self.planning_horizon, self.tran.action_dim))
        std_best_policies = tf.ones((self.planning_horizon, self.tran.action_dim))

        # perform I iterations of CEM
        for i in range(self.n_cem_policy_iterations):
            policy_distr = tfp.distributions.MultivariateNormalDiag(loc=mean_best_policies, scale_diag=std_best_policies)
            policies = policy_distr.sample([self.n_policies])
            policies = tf.clip_by_value(policies, clip_value_min=-1, clip_value_max=1)

            # project trajectory into the future using transition model and calculate FEEF for each policy
            policy_results = self.rollout_policies(policies.numpy(), latent_z)
            expected_free_energy = self.evaluate_policy(*policy_results)

            expected_free_energy = tf.convert_to_tensor(expected_free_energy)

            # sum over the time steps to get the FEEF for each policy
            expected_free_energy_sum = tf.reduce_sum(expected_free_energy, axis=0)

            # multiply by -1 to find largest value which is equivalent to smallest FEEF with top_k
            neg_sum = -1*expected_free_energy_sum

            result = tf.math.top_k(neg_sum, self.n_policy_candidates, sorted=False)
            min_indices = result.indices

            # update the policy distributions
            mean_best_policies = tf.reduce_mean(tf.gather(policies, min_indices), axis=0)
            std_best_policies = tf.math.reduce_std(tf.gather(policies, min_indices), axis=0)

        return mean_best_policies, std_best_policies


    def rollout_policies(self, policies, z_t_minus_one):
        """
        Rollout policies and compute the expected free energy of each.
        :param policies:
        :param z_t_minus_one:
        :return:
        """

        # stack up the new observation to have shape (self.n_policies, latent_dim) when z_t_minus is tensor with shape (1, latent_dim)
        prev_latent_mean = tf.squeeze(tf.stack([z_t_minus_one]*self.n_policies, axis=1))

        policy_posteriors = []
        policy_sds = []
        likelihoods = []
        z_means = []
        z_sds = []

        # get the starting hidden state that coressponds to the memory stored by the previous sequences. Should have shape (1, self.tran.num_hidden_units) for the observed sequence
        # extend the current hidden state to the number of policies present
        if self.tran_hidden_state is None:
            cur_hidden_state = np.zeros((self.n_policies, self.tran.hidden_units))
        else:
            cur_hidden_state = np.vstack([self.tran_hidden_state]*self.n_policies)

        # find the predicted latent states from the transition model
        for t in range(self.planning_horizon):

            # concatenate the latent state and action for the transition model. Then reshape to pass to GRU
            latent_plus_action = np.concatenate([prev_latent_mean, policies[:, t, :]], axis=1)
            tran_input = latent_plus_action.reshape((self.n_policies, 1, latent_plus_action.shape[1]))

            next_latent_mean, next_latent_sd, next_hidden_state, _ = self.tran((tran_input, cur_hidden_state))  # shape = [num policies, latent dim]

            # update the GRU hidden state for use with the next policies
            cur_hidden_state = next_hidden_state

            policy_posteriors.append(next_latent_mean)
            policy_sds.append(next_latent_sd)

            next_likelihoods = self.model_vae.decoder(next_latent_mean)
            likelihoods.append(next_likelihoods)

            next_posterior_means, next_posteriors_sds, next_posteriors_z = self.model_vae.encoder(next_likelihoods)
            z_means.append(next_posterior_means)
            z_sds.append(next_posteriors_sds)

            prev_latent_mean = next_latent_mean

        return policy_posteriors, policy_sds, likelihoods, z_means, z_sds


    def evaluate_policy(self, policy_posteriors, policy_sd, predicted_likelihood, predicted_posterior, predicted_posterior_sd):
        """

        :param policy_posteriors:
        :param policy_sd:
        :param predicted_likelihood:
        :param predicted_posterior:
        :param predicted_posterior_sd:
        :return:
        """

        if self.use_FEEF:
            return self.FEEF(policy_posteriors, policy_sd, predicted_likelihood, predicted_posterior, predicted_posterior_sd)
        else:
            return self.EFE(policy_posteriors, policy_sd, predicted_likelihood, predicted_posterior, predicted_posterior_sd)


    def FEEF(self, policy_posteriors_list, policy_sd_list, predicted_likelihood_list, predicted_posterior_list, predicted_posterior_sd_list):
        """

        :param policy_posteriors_list:
        :param policy_sd_list:
        :param predicted_likelihood_list:
        :param predicted_posterior_list:
        :param predicted_posterior_sd_list:
        :return:
        """

        FEEFs = []

        for t in range(self.planning_horizon):

            # extract the values for each time step
            predicted_likelihood = predicted_likelihood_list[t]
            policy_posteriors = policy_posteriors_list[t]
            policy_sd = policy_sd_list[t]
            predicted_posterior = predicted_posterior_list[t]
            predicted_posterior_sd = predicted_posterior_sd_list[t]

            # evaluate the extrinsic part of FEEF
            if self.use_efe_extrinsic:
                likelihood_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_likelihood, scale_diag=np.ones_like(predicted_likelihood))

                if self.prior_model is None:

                    # create the prior distribution
                    prior_preferences_mean = tf.convert_to_tensor(np.stack([self.given_prior_mean]*self.n_policies), dtype="float32")
                    prior_preferences_stddev = tf.convert_to_tensor(np.stack([self.given_prior_stddev]*self.n_policies), dtype="float32")

                    prior_dist = tfp.distributions.MultivariateNormalDiag(loc=prior_preferences_mean, scale_diag=prior_preferences_stddev)

                    kl_extrinsic = tfp.distributions.kl_divergence(likelihood_dist, prior_dist)

                # Compute the extrinisc approximation with the prior model
                else:
                    kl_extrinsic = 1 - self.prior_model_scaling_factor * self.prior_model(predicted_posterior)
                    kl_extrinsic = tf.reduce_sum(kl_extrinsic, axis=-1)

            # if we don't use extrinsic set it to zero
            else:
                kl_extrinsic = tf.zeros(self.n_policies, dtype="float")

            # evaluate the KL INTRINSIC part
            if self.use_kl_intrinsic:

                policy_posteriors_dist = tfp.distributions.MultivariateNormalDiag(loc=policy_posteriors, scale_diag=policy_sd)
                predicted_posterior_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_posterior, scale_diag=predicted_posterior_sd)

                kl_intrinsic = tfp.distributions.kl_divergence(predicted_posterior_dist, policy_posteriors_dist)

            else:
                kl_intrinsic = tf.zeros(self.n_policies, dtype="float")

            # combine the extrinsic and intrinsic parts for total FEEF
            FEEF = kl_extrinsic - kl_intrinsic
            FEEFs.append(FEEF)

        return FEEFs

    def EFE(self, policy_posteriors_list, policy_sd_list, predicted_likelihood_list, predicted_posterior_list, predicted_posterior_sd_list):
        """
        Compute EFE of projected policies
        :param policy_posteriors_list:
        :param policy_sd_list:
        :param predicted_likelihood_list:
        :param predicted_posterior_list:
        :param predicted_posterior_sd_list:
        :return:
        """

        EFEs = []

        for t in range(self.planning_horizon):

            # extract the values for each time step
            predicted_likelihood = predicted_likelihood_list[t]
            policy_posteriors = policy_posteriors_list[t]
            policy_sd = policy_sd_list[t]
            predicted_posterior = predicted_posterior_list[t]
            predicted_posterior_sd = predicted_posterior_sd_list[t]

            # evaluate the extrinsic EFE
            if self.use_efe_extrinsic:

                if self.prior_model is None:

                    # create the prior distribution
                    prior_preferences_mean = tf.convert_to_tensor(np.stack(self.given_prior_mean), dtype="float32")
                    prior_preferences_stddev = tf.convert_to_tensor(np.stack(self.given_prior_stddev), dtype="float32")

                    prior_dist = tfp.distributions.MultivariateNormalDiag(loc=prior_preferences_mean, scale_diag=prior_preferences_stddev)

                    # compute extrinsic prior preferences term
                    efe_extrinsic = -1 * tf.math.log(prior_dist.prob(predicted_likelihood))

                else:
                    efe_extrinsic = 1 - self.prior_model_scaling_factor * self.prior_model(predicted_posterior)
                    efe_extrinsic = tf.reduce_sum(efe_extrinsic, axis=-1)

            # if we don't use extrinsic set it to zero
            else:
                efe_extrinsic = tf.zeros(self.n_policies, dtype="float")

            # evaluate the KL INTRINSIC part
            if self.use_kl_intrinsic:

                policy_posteriors_dist = tfp.distributions.MultivariateNormalDiag(loc=policy_posteriors, scale_diag=policy_sd)
                predicted_posterior_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_posterior, scale_diag=predicted_posterior_sd)
                kl_intrinsic = tfp.distributions.kl_divergence(predicted_posterior_dist, policy_posteriors_dist)

            else:
                kl_intrinsic = tf.zeros(self.n_policies, dtype="float")

            # combine for full EFE
            EFE = efe_extrinsic - kl_intrinsic
            EFEs.append(EFE)

        return EFEs
