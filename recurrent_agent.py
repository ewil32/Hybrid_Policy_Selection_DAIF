import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# from vae_recurrent import VAE


class DAIFAgentRecurrent:

    def __init__(self,
                 prior_model,
                 vae,
                 tran,
                 given_prior_mean,
                 given_prior_stddev,
                 agent_time_ratio=6,
                 planning_horizon=15,
                 n_policies=1500,
                 n_cem_policy_iterations=2,
                 n_policy_candidates=70,
                 tran_train_epochs=1,
                 vae_train_epochs=1,
                 train_vae=True,
                 train_tran=True,
                 train_prior_model=False,
                 use_kl_extrinsic=True,
                 use_kl_intrinsic=True,
                 use_FEEF=True,
                 show_vae_training=False,
                 show_tran_training=False,
                 show_prior_training=False):

        super(DAIFAgentRecurrent, self).__init__()

        self.planning_horizon = planning_horizon
        self.n_policy_candidates = n_policy_candidates
        self.n_policies = n_policies
        self.n_cem_policy_iterations = n_cem_policy_iterations

        self.vae_train_epochs = vae_train_epochs
        self.tran_train_epochs = tran_train_epochs
        self.train_vae = train_vae
        self.train_tran = train_tran

        # do we use the kl divergence for extrinsic vs intrinsic
        self.use_kl_intrinsic = use_kl_intrinsic
        self.use_kl_extrinsic = use_kl_extrinsic

        # do we use the FEEF or EFE?
        self.use_FEEF = use_FEEF

        self.given_prior_mean = given_prior_mean
        self.given_prior_stddev = given_prior_stddev

        # full vae
        self.model_vae = vae
        self.model_vae.compile(optimizer=tf.keras.optimizers.Adam())
        self.show_vae_training = show_vae_training

        # transition
        # takes action plus last state and outputs next latent state
        self.tran = tran
        self.tran.compile(optimizer=tf.keras.optimizers.Adam())
        self.show_tran_training = show_tran_training
        # track the hidden state of the transition gru model
        self.hidden_state = None

        # Prior model
        self.prior_model = prior_model
        self.train_prior = train_prior_model
        self.show_prior_training = show_prior_training

        # how much is the agents planning time compressed compared to the simulation time
        self.agent_time_ratio = agent_time_ratio


    def train(self, pre_observations_raw, post_observations_raw, actions_complete, rewards, verbose=0):

        # compress the observations based on the agents time compression factor
        pre_observations = pre_observations_raw[::self.agent_time_ratio]  # for example take every 6th element
        post_observations = np.flip(np.flip(post_observations_raw)[::self.agent_time_ratio])  # ends up at element
        # post_observations = np.array([post_observations_raw[i] for i in range(len(post_observations_raw)) if i % self.agent_time_ratio == self.agent_time_ratio - 1])
        #
        # print(pre_observations_raw)
        # print(pre_observations)
        # print(post_observations_raw)
        # print(post_observations)

        # pre_observations = pre_observations_raw
        # post_observations = post_observations_raw

        #### TRAIN THE TRANSITION MODEL ####
        if self.train_tran:
            # only look at the first n actions that we took
            actions = actions_complete[0: len(pre_observations)]

            num_observations = pre_observations.shape[0]
            observation_dim = pre_observations.shape[1]
            action_dim = actions.shape[1]
            latent_dim = self.model_vae.latent_dim

            # action_dim = 1  # TODO fix this to allow different actions

            # find the actual observed latent states using the vae
            pre_latent_mean, pre_latent_stddev, pre_latent = self.model_vae.encoder(pre_observations)
            post_latent_mean, post_latent_stddev, post_latent = self.model_vae.encoder(post_observations)

            # set up the input training data that we use to train the transition model
            z_train = np.concatenate([np.array(pre_latent_mean), np.array(actions)], axis=1)

            # we use the sequence to find the right hidden states to use as input
            z_train_seq = z_train.reshape((1, num_observations, latent_dim + action_dim))
            z_train_singles = z_train.reshape(num_observations, 1, latent_dim + action_dim)

            # the previous hidden state is the memory after observing some sequences but it might be None
            if self.hidden_state is None:
                self.hidden_state = np.zeros((1, self.tran.hidden_units))

            # find the hidden states at t=0, t=1, t=2, ..., t=num_observations - 1
            _, _, _, h_states = self.tran((z_train_seq, self.hidden_state))

            # squeeze so we make the shape [num_observations, hidden_units]
            h_states = tf.squeeze(h_states)

            # exclude the last state as this will become the hidden state later on. next hidden state will become our new memory
            h_states_for_training = h_states[:-1]
            # next_hidden_state = h_states[-1]

            # add the current hidden state we saved to the start. This has h0, h1, h2, .. h=num_observations - 1
            h_states_for_training = tf.concat([self.hidden_state, h_states_for_training], axis=0)

            # use the hidden states with the pre and post observations to train transition model
            self.tran.fit((z_train_singles, h_states_for_training), (post_latent_mean, post_latent_stddev), epochs=self.tran_train_epochs, verbose=self.show_tran_training)

            # now find the new predicted hidden state that we will use for finding the policy
            # TODO not sure if I should pass the old hidden state or reset it to 0
            _, _, final_hidden_state, _ = self.tran((z_train_seq, self.hidden_state))
            # _, _, final_hidden_state, _ = self.tran((z_train_seq, None))

            self.hidden_state = final_hidden_state

        #### TRAIN THE VAE ####
        if self.train_vae:
            # train the vae model on post_observations because these are all new
            # self.model_vae.fit(pre_observations_raw, epochs=self.vae_train_epochs, verbose=self.show_vae_training)
            self.model_vae.fit(pre_observations, epochs=self.vae_train_epochs, verbose=self.show_vae_training)

            # print("true", pre_observations)
            # print("pred", self.model_vae(pre_observations))

        #### TRAIN THE PRIOR MODEL ####
        if self.train_prior:
            # self.prior_model.train(post_observations, rewards, verbose=self.show_prior_training)
            self.prior_model.train(post_observations_raw, rewards, verbose=self.show_prior_training)


    def select_policy(self, observation):

        # TODO do you take the mean or that latent here?
        # get the latent state from this observation
        _,  _, latent_state = self.model_vae.encoder(observation.reshape(1, observation.shape[0]))
        # latent_state = latent_state.numpy().reshape((1, latent_state.shape[0]))

        # print(latent_state)
        # select the policy
        policy_mean, policy_stddev = self.cem_policy_optimisation(latent_state)

        # return a distribution that we can sample from
        return tfp.distributions.MultivariateNormalDiag(loc=policy_mean, scale_diag=policy_stddev)


    def cem_policy_optimisation(self, z_t_minus_one):

        # need to change these two if the policy dimension changes
        mean_best_policies = tf.zeros(self.planning_horizon)
        std_best_policies = tf.ones(self.planning_horizon)

        for i in range(self.n_cem_policy_iterations):
            policy_distr = tfp.distributions.MultivariateNormalDiag(loc=mean_best_policies, scale_diag=std_best_policies)
            policies = policy_distr.sample([self.n_policies])
            policies = tf.clip_by_value(policies, clip_value_min=-1, clip_value_max=1)

            # project trajectory into the future using transition model and calculate FEEF for each policy
            policy_results = self.forward_policies(policies.numpy(), z_t_minus_one)
            FEEFs = self.evaluate_policy(*policy_results)

            FEEFs = tf.convert_to_tensor(FEEFs)

            # sum over the timesteps to get the FEEF for each policy
            FEEFs_sum = tf.reduce_sum(FEEFs, axis=0)

            # multiply by one to find largest value which is euqivalent to smallest FEEF with top_k
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

        # stack up the new observation to have shape (self.n_policies, latent_dim) when z_t_minus is tensor with shape (1, latent_dim
        prev_latent_mean = tf.squeeze(tf.stack([z_t_minus_one]*self.n_policies, axis=1))

        policy_posteriors = []
        policy_sds = []
        likelihoods = []
        z_means = []
        z_sds = []

        # get the starting hidden state that coressponds to the memory stored by the previous sequences. Should have shape (1, self.tran.num_hidden_units) for the observed sequence
        # extend the current hidden state to the number of policies present
        if self.hidden_state is None:
            cur_hidden_state = np.zeros((self.n_policies, self.tran.hidden_units))
        else:
            cur_hidden_state = np.vstack([self.hidden_state]*self.n_policies)

        # print(cur_hidden_state)

        # find the predicted latent states from the transition model
        for t in range(self.planning_horizon):

            # print(prev_latent_mean)

            ob_plus_action = np.concatenate([prev_latent_mean, policies[:, t].reshape(self.n_policies, 1)], axis=1)
            tran_input = ob_plus_action.reshape((self.n_policies, 1, ob_plus_action.shape[1]))  # reshape to pass to GRU

            # print(tran_input)

            next_latent_mean, next_latent_sd, next_hidden_state, _ = self.tran((tran_input, cur_hidden_state))  # shape = [num policies, latent dim

            # update the hidden state for use with the next policies
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
                    efe_extrinsic = self.prior_model.extrinsic_kl(predicted_likelihood)
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

            EFE = efe_extrinsic - kl_intrinsic

            EFEs.append(EFE)

        return EFEs