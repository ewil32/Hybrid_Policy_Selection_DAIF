import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

from vae import VAE


class DAIFAgentRecurrent:

    def __init__(self,
                 prior_model,
                 enc,
                 dec,
                 tran,
                 planning_horizon=15,
                 n_policies=1500,
                 n_cem_policy_iterations=2,
                 n_policy_candidates=70):

        super(DAIFAgent, self).__init__()

        self.prior_model = prior_model
        self.planning_horizon = planning_horizon
        self.n_policy_candidates = n_policy_candidates
        self.n_policies = n_policies
        self.n_cem_policy_iterations = n_cem_policy_iterations

        # encoder
        self.enc = enc

        # decoder
        # takes latent state and outputs observation
        self.dec = dec

        # full vae
        self.model_vae = VAE(enc, dec)
        self.model_vae.compile(optimizer=tf.keras.optimizers.Adam())

        # transition
        # takes action plus last state and outputs next latent state
        self.tran = tran
        self.tran.compile(optimizer=tf.keras.optimizers.Adam())


    def select_action(self, observation):

        policy_mean, policy_stddev = self.cem_policy_optimisation(observation)

        # return a distribution that we can sample from
        return tfp.distributions.MultivariateNormalDiag(loc=policy_mean, scale_diag=policy_stddev)

    def train(self, pre_observations, post_observations, actions):

        # pre and post should have shape [sim_steps, ob_dim], actions has shape [sim_steps, action_dim]

        # use vae to get latent obs
        pre_latent_mean, pre_latent_stddev = self.model_vae.encoder(pre_observations)
        post_latent_mean, post_latent_stddev = self.model_vae.encoder(post_observations)

        # use latent obs to train transition
        z_train = np.concatenate([np.array(pre_latent_mean), np.array(actions)], axis=1)
        self.tran.fit((z_train, what_is_the_initial_state), (post_latent_mean, post_latent_stddev))

        # now find the new predicted post_latents
        pred_post_latent, pred_post_stddev = self.tran((z_train, what_is_the_initial_state))

        # use hidden states from transition to regularise fitting process of vae

        #
        pass


    def train_vae(self, observation, verbose=0):
        self.model_vae.fit(observation, verbose=verbose)


    def train_transition(self, o_t_minus_one, o_t, action_t_minus_one, verbose=0):

        # find the latent reps with the decoder
        z_t_minus_1_mean, z_t_minus_1_stddev, z_t_minus = self.enc(o_t_minus_one)
        z_t_mean, z_t_stddev, z_t = self.enc(o_t)

        # concatenate action and observation for input into transition
        z_train = np.concatenate([np.array(z_t_minus_1_mean), np.array(action_t_minus_one)], axis=1)

        # train the transition model
        self.tran.fit(z_train, (z_t_mean, z_t_stddev), epochs=1, verbose=verbose)


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

        # find the predicted latent states from the transition model
        for t in range(self.planning_horizon):

            tran_input = np.concatenate([prev_latent_mean, policies[:, t].reshape(self.n_policies, 1)], axis=1)
            next_latent_mean, next_latent_sd = self.tran(tran_input)  # shape = [num policies, latent dim

            policy_posteriors.append(next_latent_mean)
            policy_sds.append(next_latent_sd)

            next_likelihoods = self.dec(next_latent_mean)
            likelihoods.append(next_likelihoods)

            next_posterior_means, next_posteriors_sds, next_posteriors_z = self.enc(next_likelihoods)
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

                # TODO how exactly is the prior defined
                # create the prior distribution
                prior_preferences = tf.convert_to_tensor(np.stack([[0.5, 100]]*self.n_policies), dtype="float32")

                prior_dist = tfp.distributions.MultivariateNormalDiag(loc=prior_preferences, scale_diag=np.ones_like(prior_preferences))

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
