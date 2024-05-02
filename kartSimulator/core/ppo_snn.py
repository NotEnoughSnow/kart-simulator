import gym
import numpy as np
import torch, torch.nn as nn
import time

from kartSimulator.core.snn_network import SNN
from kartSimulator.core import snn_utils
from torch.distributions import MultivariateNormal
import snntorch.functional as SF

class SNNPPO:

    def __init__(self, env, **hyperparameters):
        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)
        assert (type(env.action_space) == gym.spaces.Box)


        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "mps") if torch.backends.mps.is_available() else torch.device("cpu")

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # TODO num steps
        # Define the number of time steps for the simulation
        self.num_steps = 50

        # Initialize actor and critic networks
        self.actor = SNN(self.obs_dim, self.act_dim, self.num_steps).to(self.device)  # ALG STEP 1
        self.critic = SNN(self.obs_dim, 1, self.num_steps).to(self.device)

        # TODO betas
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")

        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        while t_so_far < total_timesteps:  # ALG STEP 2

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            batch_obs_ts = snn_utils.encode_to_spikes_batched(batch_obs, num_steps=self.num_steps)

            # Calculate the sum of the lengths of the trajectories
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Log the timesteps and iterations
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Evaluate the batch using the SNN
            V, _ = self.evaluate(batch_obs_ts, batch_acts)

            # Detach V from the graph to prevent gradients from flowing into the critic
            V = V.detach()

            # Calculate the advantage
            A_k = batch_rtgs - V

            # Normalize the advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Update the network for a number of epochs
            for _ in range(self.n_updates_per_iteration):
                # Re-evaluate the batch to get the current log probabilities and values
                V, curr_log_probs = self.evaluate(batch_obs_ts, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger['actor_losses'].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './snn_ppo_actor.pth')
                torch.save(self.critic.state_dict(), './snn_ppo_critic.pth')

        torch.save(self.actor.state_dict(), './ppo_actor.pth')
        torch.save(self.critic.state_dict(), './ppo_critic.pth')

        print("saved models")


    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode

            obs = self.env.reset()[0]
            truncated = False
            terminated = False


            # TODO understand vars timesteps per ep, batch, ep, timesteps, ect..
            for ep_t in range(self.max_timesteps_per_episode):

                t += 1  # Increment timesteps ran this batch so far

                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                # FIXME actions are not in range(-1,1)
                action, log_prob = self.get_action(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action.squeeze())

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # print(rew)

                # If the environment tells us the episode is terminated, break
                if terminated or truncated:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

            # Reshape data as tensors in the shape specified in function description, before returning
            batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
            batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float).squeeze()
            batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float).squeeze()
            batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

            # Log the episodic returns and episodic lengths in this batch.
            self.logger['batch_rews'] = batch_rews
            self.logger['batch_lens'] = batch_lens

            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def evaluate(self, batch_obs_ts, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)

            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
                :param batch_acts:
                :param batch_obs_ts:
        """

        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V_ts = self.critic(batch_obs_ts)[0]

        V = snn_utils.decode_first_spike_batched(V_ts).squeeze().requires_grad_(True)

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean_ts = self.actor(batch_obs_ts)[0].detach()

        mean = snn_utils.decode_first_spike_batched(mean_ts)

        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts).requires_grad_(True)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """

        obs = torch.tensor(obs)

        obs_st = snn_utils.encode_to_spikes_batched(obs.unsqueeze(0), num_steps=self.num_steps)

        # Query the actor network for a mean action
        mean_st = self.actor(obs_st)[0].detach()

        # Convert spike train outputs to action vector
        mean = snn_utils.decode_first_spike_batched(mean_st)

        # Create a distribution with the mean action and std from the covariance matrix above.
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
