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
                 train_after_exploring=True,
                 use_kl_extrinsic=True,
                 use_kl_intrinsic=True,
                 use_FEEF=True,
                 use_fast_thinking=False,
                 uncertainty_tolerance=0.05,
                 habit_model_type="name_of_model",
                 min_rewards_needed_to_train_prior=0):

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
        self.train_after_exploring = train_after_exploring

        # do we use the kl divergence for extrinsic vs intrinsic
        self.use_kl_intrinsic = use_kl_intrinsic
        self.use_kl_extrinsic = use_kl_extrinsic

        # do we use the FEEF or EFE?
        self.use_FEEF = use_FEEF

        # given prior values
        self.given_prior_mean = given_prior_mean
        self.given_prior_stddev = given_prior_stddev

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
        # self.tran_hidden_state = None
        # self.tran_hidden_state_pre_exploring = None
        # self.prev_tran_hidden_state = None

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

        self.use_fast_thinking = use_fast_thinking
        self.habit_model_type = habit_model_type
        self.uncertainty_tolerance = uncertainty_tolerance
        self.num_fast_thinking_choices = 0

        # Normally 0 but this is just a bad parameter and I don't know if it should exist
        self.min_rewards_needed_to_train_prior = min_rewards_needed_to_train_prior


    def perceive_and_act(self, observation, reward, done):
        """
        The function called to have the agent interact with the environment
        We assume the agent gets a transformed/noisy observation from the environment and then returns an action

        TODO: possibly the agent returns some other information for logging and showing experiments

        :param observation:
        :param reward:
        :param done:
        :return:
        """

        # track the world time scale observation sequence
        # self.env_time_scale_observations.append(observation)

        # if the episode is finished, then do any training on the full data set
        if done and self.train_with_replay:

            print("training on full data")
            print(self.num_fast_thinking_choices, len(self.full_action_sequence))

            # add the final observation and reward we observed to the sequences
            self.full_observation_sequence.append(observation)
            self.full_reward_sequence.append(reward)

            # Call the training function on the observation sequences to train everything we need to train
            self.train_models(np.vstack(self.full_observation_sequence),
                              np.vstack(self.full_action_sequence),
                              np.array(self.full_reward_sequence))


        # Otherwise are we at a point where we can reconsider our policy and maybe train the world model
        elif self.time_step % self.agent_time_ratio == 0:

            # add the observation to the sequence
            self.full_observation_sequence.append(observation)

            # add the reward only if it's not the first observation
            if self.time_step != 0:
                self.full_reward_sequence.append(reward)

            # We only update the model during the episode when we were exploring using the planning method and we have executed all of the actions in the policy
            if self.exploring and len(self.policy_left_to_execute) == 0:

                # print("f", self.full_observation_sequence)
                # print("e", self.full_observation_sequence[-1*(self.actions_to_execute_when_exploring + 1):])

                if self.train_after_exploring:

                    # the actions done while exploring were the last self.actions_to_execute_when_exploring
                    self.exploring_action_sequence = self.full_action_sequence[-1*self.actions_to_execute_when_exploring:]
                    self.exploring_reward_sequence = self.full_reward_sequence[-1*self.actions_to_execute_when_exploring:]
                    self.exploring_observation_sequence = self.full_observation_sequence[-1*(self.actions_to_execute_when_exploring + 1):]

                    # Call the training function on the observation sequences to train everything we need to train
                    self.train_models(np.vstack(self.exploring_observation_sequence),
                                      np.vstack(self.exploring_action_sequence),
                                      np.array(self.exploring_reward_sequence))

                # now we're no longer exploring
                self.exploring = False


            # Predict the expected observation based on the previously executed action
            action_as_array = np.array(self.action_being_executed).reshape(1, self.tran.action_dim)
            expected_observation = self.predict_next_observation(self.previous_observation, action_as_array)

            # Now we select our action. If we aren't exploring then either we act out of habit or we might need to explore
            # I think I can check this based on whether or not there are actions left to execute in the current policy
            if not self.exploring:

                # first observation will have no previous observation
                if self.use_fast_thinking and self.previous_observation is None:
                    # self.policy_left_to_execute = self.habit_action_model(observation)
                    self.policy_left_to_execute = self.select_fast_thinking_policy(observation)
                    self.policy_left_to_execute = self.policy_left_to_execute.numpy().tolist()  # tf tensor to list
                    print("fast thinking")

                # TDOD Fix this to work however it needs to
                # we need to see what the generative model now thinks about what the expected current observation is
                elif self.use_fast_thinking and np.allclose(observation, expected_observation, atol=self.uncertainty_tolerance):  # within some tolerance

                    self.policy_left_to_execute = self.select_fast_thinking_policy(observation)
                    # self.policy_left_to_execute = self.policy_left_to_execute + np.random.normal(0, scale=self.habit_action_model.action_std_dev)
                    self.policy_left_to_execute = self.policy_left_to_execute.numpy().tolist()

                    # self.tran_hidden_state = next_tran_hidden_state
                    self.num_fast_thinking_choices += 1
                    print("fast thinking")

                # the generative model is surprised so we should use the slow deliberation for planning out a policy that balances exploration and exploitation
                else:
                    # TODO should we actually sample here?
                    # print("slow thinking")
                    policy = self.select_policy(observation)
                    policy = policy.mean().numpy()
                    policy = policy.reshape(policy.shape[0], self.tran.action_dim).tolist()
                    self.policy_left_to_execute = policy[0: self.actions_to_execute_when_exploring]

                    self.exploring = True

            # finally update the previous observation and action to be the one we just had/did
            self.previous_observation = observation
            self.action_being_executed = self.policy_left_to_execute[0]
            self.full_action_sequence.append(self.action_being_executed)
            self.policy_left_to_execute.pop(0)

        # final updates increment the current timestep and return the action specified by the policy
        self.time_step += 1

        return self.action_being_executed


    def predict_next_observation(self, obs, action):

        # TODO: Fix this with the transition hidden states
        if obs is None:
            return None, None
        else:
            z_mean, z_std, z = self.model_vae.encoder(obs)
            # print(z_mean.shape)
            # print(action.shape)
            z_mean = z_mean.numpy()
            z_plus_action = np.concatenate([z_mean, action], axis=1)
            # print(z_mean)
            # print(action)
            # print(z_plus_action)
            # print(z_plus_action.shape)
            # z_plus_action = z_plus_action.reshape(1, 1, z_plus_action.shape[1])
            # print(z_plus_action)

            next_latent_mean, next_latent_sd = self.tran(z_plus_action)
            # print(next_latent_mean)
            next_observation = self.model_vae.decoder(next_latent_mean)
            # print(next_observation)
            return next_observation.numpy()


    def reset_all_states(self):
        self.time_step = 0
        self.exploring = False

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
        self.previous_action_executed = None

        self.num_fast_thinking_choices = 0


    def train_models(self, observations_full, actions, rewards):

        pre_observations = observations_full[:-1]
        post_observations = observations_full[1:]

        # find the actual observed latent states using the vae
        pre_latent_mean, pre_latent_stddev, pre_latent = self.model_vae.encoder(pre_observations)
        post_latent_mean, post_latent_stddev, post_latent = self.model_vae.encoder(post_observations)

        #### TRAIN THE TRANSITION MODEL ####
        if self.train_tran:

            # set up the input training data that we use to train the transition model
            z_train = np.concatenate([np.array(pre_latent_mean), actions], axis=1)
            self.tran.fit(z_train, (post_latent_mean, post_latent_stddev), epochs=self.tran.train_epochs, verbose=self.tran.show_training, batch_size=pre_observations.shape[0])


        #### TRAIN THE VAE ####
        if self.train_vae:
            # train the vae model on post_observations because these are all new
            # self.model_vae.fit(pre_observations_raw, epochs=self.vae_train_epochs, verbose=self.show_vae_training)
            self.model_vae.fit(pre_observations, epochs=self.model_vae.train_epochs, verbose=self.model_vae.show_training, batch_size=pre_observations.shape[0])


        #### TRAIN THE PRIOR MODEL ####
        # TODO fix how this part should work
        if self.train_prior:
            # self.prior_model.train(post_observations, rewards, verbose=self.show_prior_training)
            if max(rewards) > self.min_rewards_needed_to_train_prior:
                # self.prior_model.train(post_observations, rewards)
                self.prior_model.train(post_latent_mean, rewards)


        #### TRAIN THE HABIT ACTION NET ####
        if self.train_habit_net:

            if self.habit_model_type == "PG":

                # TODO I think for the final state the V(s_t+1) should be set to 0
                # ADVANTAGE
                v_state = self.prior_model(pre_latent_mean)
                v_plus_one_state = self.prior_model(post_latent_mean)
                advantage = rewards + self.prior_model.discount_factor * v_plus_one_state - v_state

                # print(advantage)

                # DDPG and policy gradient interface with same function
                # self.habit_action_model.train(pre_latent_mean, actions, rewards_to_train_on, post_latent_mean)
                self.habit_action_model.train(pre_latent_mean, actions, advantage, post_latent_mean)

            if self.habit_model_type == "DDPG":
                self.habit_action_model.train(pre_latent_mean, actions, rewards, post_latent_mean)


    def select_fast_thinking_policy(self, observation):

        # TODO should you select the mean here?
        # _,  _, latent_state = self.model_vae.encoder(observation)
        latent_state,  _, _ = self.model_vae.encoder(observation)
        action = self.habit_action_model.select_action(latent_state)

        return action


    def select_policy(self, observation):
        """
        :param observation: needs to be [n, observation_dim] shape np array or tf tensor
        :return:
        """

        # TODO do you take the mean or that latent here?
        # get the latent state from this observation
        # TODO should I use the mean here?
        _,  _, latent_state = self.model_vae.encoder(observation)
        # latent_state,  _, _ = self.model_vae.encoder(observation)
        # latent_state = latent_state.numpy().reshape((1, latent_state.shape[0]))
        # print(latent_state)
        # print(latent_state)
        # select the policy
        policy_mean, policy_stddev = self.cem_policy_optimisation(latent_state)

        # return a distribution that we can sample from
        return tfp.distributions.MultivariateNormalDiag(loc=policy_mean, scale_diag=policy_stddev)


    # TODO Fix this so we can use different action dimensions
    def cem_policy_optimisation(self, latent_z):

        # need to change these two if the policy dimension changes
        mean_best_policies = tf.zeros((self.planning_horizon, self.tran.action_dim))
        std_best_policies = tf.ones((self.planning_horizon, self.tran.action_dim))

        # print(mean_best_policies)
        # print(mean_best_policies.shape)

        for i in range(self.n_cem_policy_iterations):
            policy_distr = tfp.distributions.MultivariateNormalDiag(loc=mean_best_policies, scale_diag=std_best_policies)
            policies = policy_distr.sample([self.n_policies])
            # print("p", policies.shape)
            policies = tf.clip_by_value(policies, clip_value_min=-1, clip_value_max=1)
            # policies = tf.clip_by_value(policies, clip_value_min=-1, clip_value_max=1)

            # project trajectory into the future using transition model and calculate FEEF for each policy
            policy_results = self.forward_policies(policies.numpy(), latent_z)
            FEEFs = self.evaluate_policy(*policy_results)

            # print("POLICIES", policies)
            # print("FEEFS", FEEFs)

            FEEFs = tf.convert_to_tensor(FEEFs)

            # sum over the timesteps to get the FEEF for each policy
            FEEFs_sum = tf.reduce_sum(FEEFs, axis=0)

            # multiply by -1 to find largest value which is euqivalent to smallest FEEF with top_k
            neg_FEEF_sum = -1*FEEFs_sum

            result = tf.math.top_k(neg_FEEF_sum, self.n_policy_candidates, sorted=False)
            min_FEEF_indices = result.indices

            # update the policy distributions
            mean_best_policies = tf.reduce_mean(tf.gather(policies, min_FEEF_indices), axis=0)
            std_best_policies = tf.math.reduce_std(tf.gather(policies, min_FEEF_indices), axis=0)


        # TODO not sure why we need all of this is with the x means? I think it's for training but maybe not

        # One last forward pass to gather the stats of the policy mean
        #FEEFs, next_x_means, next_x_stds = self._forward_policies(mean_best_policies.unsqueeze(1))
        # return mean_best_policies, std_best_policies, FEEFs.detach().squeeze(1), next_x_means.detach().squeeze(1), next_x_stds.detach().squeeze(1)

        return mean_best_policies, std_best_policies


    def forward_policies(self, policies, z_t_minus_one):
        """
        Forward propogate a policy and compute the FEEF of each policy
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

        # find the predicted latent states from the transition model
        for t in range(self.planning_horizon):

            # print(prev_latent_mean)
            # print(policies[:, t, :].shape)
            ob_plus_action = np.concatenate([prev_latent_mean, policies[:, t, :]], axis=1)
            # print(ob_plus_action.shape)
            # tran_input = ob_plus_action.reshape((self.n_policies, 1, ob_plus_action.shape[1]))  # reshape to pass to GRU

            # print(tran_input.shape)

            next_latent_mean, next_latent_sd = self.tran(ob_plus_action)  # shape = [num policies, latent dim]

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

        if self.use_FEEF:
            return self.FEEF(policy_posteriors, policy_sd, predicted_likelihood, predicted_posterior, predicted_posterior_sd)
        else:
            return self.EFE(policy_posteriors, policy_sd, predicted_likelihood, predicted_posterior, predicted_posterior_sd)


    def FEEF(self, policy_posteriors_list, policy_sd_list, predicted_likelihood_list, predicted_posterior_list, predicted_posterior_sd_list):
        """
        Compute the FEEF for policy selection
        :param policy_posteriors:
        :param predicted_likelihood:
        :param predicted_posterior:
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

            # !!!! evaluate the EXTRINSIC KL divergence !!!!

            # convert to normal distributions
            # TODO Why is the stddev 1s here? I think because we assume it is on the true state of the world.

            if self.use_kl_extrinsic:
                likelihood_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_likelihood, scale_diag=np.ones_like(predicted_likelihood))

                if self.prior_model is None:

                    # TODO how exactly is the prior defined? After you apply transformations what is the prior
                    # create the prior distribution
                    prior_preferences_mean = tf.convert_to_tensor(np.stack([self.given_prior_mean]*self.n_policies), dtype="float32")
                    prior_preferences_stddev = tf.convert_to_tensor(np.stack([self.given_prior_stddev]*self.n_policies), dtype="float32")

                    prior_dist = tfp.distributions.MultivariateNormalDiag(loc=prior_preferences_mean, scale_diag=prior_preferences_stddev)

                    kl_extrinsic = tfp.distributions.kl_divergence(likelihood_dist, prior_dist)

                # Compute the extrinisc approximation with the prior model
                else:
                    kl_extrinsic = self.prior_model.extrinsic_kl(predicted_likelihood)
                    kl_extrinsic = tf.reduce_sum(kl_extrinsic, axis=-1)

            # if we don't use extrinsic set it to zero
            else:
                kl_extrinsic = tf.zeros(self.n_policies, dtype="float")

            # !!!! evaluate the KL INTRINSIC part !!!!
            if self.use_kl_intrinsic:

                policy_posteriors_dist = tfp.distributions.MultivariateNormalDiag(loc=policy_posteriors, scale_diag=policy_sd)
                predicted_posterior_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_posterior, scale_diag=predicted_posterior_sd)

                kl_intrinsic = tfp.distributions.kl_divergence(predicted_posterior_dist, policy_posteriors_dist)

            else:
                kl_intrinsic = tf.zeros(self.n_policies, dtype="float")

            # print("Extrinsic", kl_extrinsic)
            # print("Intrinsic", kl_intrinsic)

            FEEF = kl_extrinsic - kl_intrinsic

            FEEFs.append(FEEF)

        return FEEFs


    # TODO Find out how this works with the log probability extrinsic term
    def EFE(self, policy_posteriors_list, policy_sd_list, predicted_likelihood_list, predicted_posterior_list, predicted_posterior_sd_list):
        """
        Compute the EFE for policy selection
        :param policy_posteriors:
        :param predicted_likelihood:
        :param predicted_posterior:
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

            # !!!! evaluate the EXTRINSIC KL divergence !!!!

            # convert to normal distributions
            # TODO Why is the stddev 1s here? I think because we assume it is on the true state of the world.

            if self.use_kl_extrinsic:
                likelihood_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_likelihood, scale_diag=np.ones_like(predicted_likelihood))

                if self.prior_model is None:

                    # TODO how exactly is the prior defined? After you apply transformations what is the prior
                    # create the prior distribution
                    prior_preferences_mean = tf.convert_to_tensor(np.stack(self.given_prior_mean), dtype="float32")
                    prior_preferences_stddev = tf.convert_to_tensor(np.stack(self.given_prior_stddev), dtype="float32")

                    prior_dist = tfp.distributions.MultivariateNormalDiag(loc=prior_preferences_mean, scale_diag=prior_preferences_stddev)

                    # compute extrinsic prior preferences term
                    efe_extrinsic = -1 * tf.math.log(prior_dist.prob(predicted_likelihood))

                # TODO Can I use the learned prior model here?
                else:
                    # efe_extrinsic = self.prior_model.extrinsic_kl(predicted_likelihood)
                    efe_extrinsic = self.prior_model.extrinsic_kl(predicted_posterior)
                    efe_extrinsic = tf.reduce_sum(efe_extrinsic, axis=-1)

            # if we don't use extrinsic set it to zero
            else:
                efe_extrinsic = tf.zeros(self.n_policies, dtype="float")

            # !!!! evaluate the KL INTRINSIC part !!!!
            if self.use_kl_intrinsic:

                policy_posteriors_dist = tfp.distributions.MultivariateNormalDiag(loc=policy_posteriors, scale_diag=policy_sd)
                predicted_posterior_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_posterior, scale_diag=predicted_posterior_sd)

                kl_intrinsic = tfp.distributions.kl_divergence(predicted_posterior_dist, policy_posteriors_dist)

            else:
                kl_intrinsic = tf.zeros(self.n_policies, dtype="float")

            # print("EX")
            # print(efe_extrinsic)
            # print("IN")
            # print(kl_intrinsic)

            EFE = efe_extrinsic - kl_intrinsic

            EFEs.append(EFE)

        return EFEs