import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from basic_agent.vae import VAE


class DAIFAgentRecurrent:

    def __init__(self,
                 prior_model,
                 vae,
                 tran,
                 given_prior_mean,
                 given_prior_stddev,
                 planning_horizon=15,
                 n_policies=1500,
                 n_cem_policy_iterations=2,
                 n_policy_candidates=70):

        super(DAIFAgentRecurrent, self).__init__()

        self.prior_model = prior_model
        self.planning_horizon = planning_horizon
        self.n_policy_candidates = n_policy_candidates
        self.n_policies = n_policies
        self.n_cem_policy_iterations = n_cem_policy_iterations

        self.vae_train_epochs = 2
        self.tran_train_epochs = 2

        self.given_prior_mean = given_prior_mean
        self.given_prior_stddev = given_prior_stddev

        # full vae
        self.model_vae = vae
        self.model_vae.compile(optimizer=tf.keras.optimizers.Adam())

        # transition
        # takes action plus last state and outputs next latent state
        self.tran = tran
        self.tran.compile(optimizer=tf.keras.optimizers.Adam())

        self.hidden_state = None


    def select_action(self, observation):

        policy_mean, policy_stddev = self.cem_policy_optimisation(observation)

        # return a distribution that we can sample from
        return tfp.distributions.MultivariateNormalDiag(loc=policy_mean, scale_diag=policy_stddev)


    def train(self, pre_observations, post_observations, actions, verbose=0):

        # pre and post should have shape [sim_steps, ob_dim], actions has shape [sim_steps, action_dim]

        # use vae to get latent obs
        pre_latent_mean, pre_latent_stddev, pre_latent = self.model_vae.encoder(pre_observations)
        post_latent_mean, post_latent_stddev, post_latent = self.model_vae.encoder(post_observations)

        # use latent obs to train transition
        z_train = np.concatenate([np.array(pre_latent_mean), np.array(actions)], axis=1)

        # 1 example 12 sim steps, 3 ob dim
        z_train_seq = z_train.reshape((1, 12, 3))
        z_train_singles = z_train.reshape(12, 1, 3)

        if self.hidden_state is None:
            self.hidden_state = np.zeros((1, self.tran.hidden_units))

        # get the hidden states for the sequences
        _, _, _, h_states = self.tran((z_train_seq, self.hidden_state))

        # hidden states for t=0, t=1, ..., t=n-1 and we want to exclude the last one
        h_states_to_use = h_states.numpy().reshape(12, self.tran.hidden_units)[:-1]

        # use the hidden states as memory for inputting individual sequences
        hidden_states_for_training = np.vstack([self.hidden_state, h_states_to_use])
        self.tran.fit((z_train_singles, hidden_states_for_training), (post_latent_mean, post_latent_stddev), epochs=self.tran_train_epochs, verbose=verbose)

        # now find the new predicted post_latents
        pred_post_latent_mean, pred_post_stddev, _, _ = self.tran((z_train_singles, hidden_states_for_training))

        # use hidden states from transition to regularise fitting process of vae
        # reg_dist = tfp.distributions.MultivariateNormalDiag(loc=pred_post_latent_mean, scale_diag=pred_post_stddev)

        self.model_vae.fit(post_observations, (pred_post_latent_mean, pred_post_stddev), epochs=self.vae_train_epochs, verbose=verbose)

        # set the hidden state to use in the next training step
        _, _, final_hidden_state, _ = self.tran((z_train_seq, self.hidden_state))

        self.hidden_state = final_hidden_state


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

        # stack up the new observation to have shape [self.n_policies, len(z_t_minus_one)]
        prev_latent_mean = np.stack([z_t_minus_one]*self.n_policies)

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

        # find the predicted latent states from the transition model
        for t in range(self.planning_horizon):

            ob_plus_action = np.concatenate([prev_latent_mean, policies[:, t].reshape(self.n_policies, 1)], axis=1)
            tran_input = ob_plus_action.reshape((self.n_policies, 1, ob_plus_action.shape[1]))  # reshape to pass to GRU

            next_latent_mean, next_latent_sd, next_hidden_state, _ = self.tran((tran_input, cur_hidden_state))  # shape = [num policies, latent dim

            # update the hidden state for use with the next policies
            cur_hidden_state = next_hidden_state

            policy_posteriors.append(next_latent_mean)
            policy_sds.append(next_latent_sd)

            next_likelihoods = self.dec(next_latent_mean)
            likelihoods.append(next_likelihoods)

            next_posterior_means, next_posteriors_sds, next_posteriors_z = self.model_vae.encoder(next_likelihoods)
            z_means.append(next_posterior_means)
            z_sds.append(next_posteriors_sds)

            prev_latent_mean = next_latent_mean

        return policy_posteriors, policy_sds, likelihoods, z_means, z_sds


    def evaluate_policy(self, policy_posteriors, policy_sd, predicted_likelihood, predicted_posterior, predicted_posterior_sd):

        return self.FEEF(policy_posteriors, policy_sd, predicted_likelihood, predicted_posterior, predicted_posterior_sd)


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
            likelihood_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_likelihood, scale_diag=np.ones_like(predicted_likelihood))

            if self.prior_model is None:

                # TODO how exactly is the prior defined? After you apply transformations what is the prior
                # create the prior distribution
                prior_preferences_mean = tf.convert_to_tensor(np.stack([self.given_prior_mean]*self.n_policies), dtype="float32")
                prior_preferences_stddev = tf.convert_to_tensor(np.stack([self.given_prior_stddev]*self.n_policies), dtype="float32")

                prior_dist = tfp.distributions.MultivariateNormalDiag(loc=prior_preferences_mean, scale_diag=prior_preferences_stddev)

            # TODO Fix the learned prior model
            else:
                prior_dist = self.prior_model()

            kl_extrinsic = tfp.distributions.kl_divergence(likelihood_dist, prior_dist)

            # !!!! evaluate the KL INTRINSIC part !!!!
            policy_posteriors_dist = tfp.distributions.MultivariateNormalDiag(loc=policy_posteriors, scale_diag=policy_sd)
            predicted_posterior_dist = tfp.distributions.MultivariateNormalDiag(loc=predicted_posterior, scale_diag=predicted_posterior_sd)

            kl_intrinsic = tfp.distributions.kl_divergence(predicted_posterior_dist, policy_posteriors_dist)

            FEEF = kl_extrinsic - kl_intrinsic

            FEEFs.append(FEEF)

        return FEEFs


    def EFE(self, policy_posteriors, predicted_likelihood, predicted_posterior):
        """
        Compute the EFE for policy selection
        :param policy_posteriors:
        :param predicted_likelihood:
        :param predicted_posterior:
        :return:
        """
        pass
