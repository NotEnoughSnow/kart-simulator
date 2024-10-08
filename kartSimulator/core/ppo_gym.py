import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

import h5py

from kartSimulator.core.actor_network import ActorNetwork
from kartSimulator.core.critic_network import CriticNetwork
from kartSimulator.core.standard_network import FFNetwork


class PPO_gym:

    def __init__(self, env, record_ghost, save_model, record_tb, save_dir, **hyperparameters):

        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)
        assert (type(env.action_space) == gym.spaces.Box)

        self.record_tb = record_tb
        self.record_ghost = record_ghost
        self.save_model = save_model

        self.run_directory = save_dir

        print(f"Saving to '{self.run_directory}' after training")

        if self.record_tb:
            print("Recording tensorboard data")
            self.writer = SummaryWriter(self.run_directory)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        print(f"obs shape :{env.observation_space.shape} \n"
              f"action shape :{env.action_space.shape}")

        # Initialize actor and critic networks
        # self.actor = ActorNetwork(self.obs_dim, self.act_dim)  # ALG STEP 1
        # self.critic = CriticNetwork(self.obs_dim, 1)

        self.actor = FFNetwork(self.obs_dim, self.act_dim)  # ALG STEP 1
        self.critic = FFNetwork(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # TODO values
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
            'lr': [],
        }

    def learn(self, total_timesteps):

        print(
            f"Learning... Running {self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps",
            end='')

        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        total_ghost_ep = []
        total_ghost_ts = []

        while t_so_far < total_timesteps:

            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_ghosts = self.rollout()
            # batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_ghosts = self.rollout()

            self.logger['batch_delta_t'] = time.time_ns()

            total_ghost_ep.append(batch_ghosts)
            total_ghost_ts.append(batch_lens)

            # Calculate advantage at k-th iteration
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss_arr = []

            # print("batch ", step)
            # print("mini batch ", minibatch_size)

            explained_variance = 1 - torch.var(batch_rtgs - V) / torch.var(batch_rtgs)
            if self.record_tb:
                self.writer.add_scalar('train/explained_variance', explained_variance, self.logger['t_so_far'])

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.logger['lr'] = new_lr

                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t)
                    V, curr_log_probs, dist, entropy_loss = self.evaluate(mini_obs, mini_acts)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    # For why we use log probabilities instead of actual probabilities,
                    # here's a great explanation:
                    # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # TL;DR makes gradient ascent easier behind the scenes.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)

                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.

                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    actor_loss = actor_loss + self.ent_coef * entropy_loss
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    clipped = (ratios < 1 - self.clip) | (ratios > 1 + self.clip)

                    clip_fraction = torch.mean(clipped.float())

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    # calculating metrics
                    total_loss = actor_loss + critic_loss
                    policy_gradient_loss = actor_loss
                    value_loss = critic_loss

                    loss_arr.append(actor_loss.detach())

                    # Approximating KL Divergence
                    if self.target_kl is not None and approx_kl > self.target_kl:
                        break  # if kl aboves threshold
                    if self.record_tb:
                        self.writer.add_scalar('train/clip_range', self.clip, self.logger['t_so_far'])
                        self.writer.add_scalar('train/clip_fraction', clip_fraction, self.logger['t_so_far'])
                        self.writer.add_scalar('train/approx_kl', approx_kl, self.logger['t_so_far'])
                        self.writer.add_scalar('train/policy_gradient_loss', policy_gradient_loss,
                                               self.logger['t_so_far'])
                        self.writer.add_scalar('train/value_loss', value_loss, self.logger['t_so_far'])
                        self.writer.add_scalar('train/loss', total_loss, self.logger['t_so_far'])

                # Log actor loss
                avg_loss = sum(loss_arr) / len(loss_arr)
                self.logger['actor_losses'].append(avg_loss)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if self.save_model:
                if i_so_far % self.save_freq == 0:
                    print("Quick-saving models")
                    torch.save(self.actor.state_dict(), f'{self.run_directory}/ppo_actor.pth')
                    torch.save(self.critic.state_dict(), f'{self.run_directory}/ppo_critic.pth')

        # model_save_directory = utils.get_next_run_directory(f'./saved_models_base/',self.experiment_type)

        if self.save_model:
            print("Saving actor & critic models")
            torch.save(self.actor.state_dict(), f'{self.run_directory}/ppo_actor.pth')
            torch.save(self.critic.state_dict(), f'{self.run_directory}/ppo_critic.pth')

        print("Finished successfully!")

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.gae_lambda * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)

    def rollout(self):

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_dones = []
        batch_vals = []

        batch_ghosts = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode

            ep_dones = []
            ep_vals = []
            ghost_ep = []

            obs = self.env.reset()[0]
            truncated = False
            terminated = False
            done = False

            ep_t = 0

            while not done:

                ep_dones.append(done)

                t += 1  # Increment timesteps ran this batch so far

                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                # FIXME actions are not in range(-1,1)
                action, log_prob = self.get_action(obs)
                val = self.critic(obs)

                obs, rew, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated

                # print("fps", info["fps"])
                # print("position", info["position"])

                ghost_ep.append((0,0))

                # TODO simplify batch actions and ghost data
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_vals.append(val.flatten())

                # print(rew)

                # TODO add fps
                if self.record_tb:
                    self.writer.add_scalar('time/fps', 0, self.logger['t_so_far'])

                ep_t += 1

            batch_ghosts.append(ghost_ep)

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

            ep_rew_mean = np.sum(ep_rews)
            # print(ep_rew_mean)

            if self.record_tb:
                self.writer.add_scalar('rollout/ep_rew_mean', ep_rew_mean, self.logger['t_so_far'])
                self.writer.add_scalar('rollout/ep_len_mean', (ep_t + 1), self.logger['t_so_far'])

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        # batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_ghosts

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        # mean, log_std = self.actor(obs)
        # std = log_std.exp()

        # cov_mat = torch.diag_embed(std ** 2)

        mean = self.actor(obs)
        cov_mat = self.cov_mat

        # Create a distribution with the mean action and std. We use torch.diag to create
        # a diagonal covariance matrix from the std.
        dist = MultivariateNormal(mean, cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
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
        """

        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        # mean, log_std = self.actor(batch_obs)
        # std = log_std.exp()

        # cov_mat = torch.diag_embed(std ** 2)

        mean = self.actor(batch_obs)

        cov_mat = self.cov_mat

        dist = MultivariateNormal(mean, cov_mat)

        entropy_loss = -dist.entropy().mean()

        if self.record_tb:
            self.writer.add_scalar('train/entropy_loss', entropy_loss, self.logger['t_so_far'])
            # self.writer.add_scalar('train/std', std.mean(), self.logger['t_so_far'])

        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist, entropy_loss


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
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.ent_coef = 0
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.num_minibatches = 8
        self.gae_lambda = 0.95

        # Miscellaneous parameters
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

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
        rollout_time = (self.logger['batch_delta_t'] - self.logger['delta_t']) / 1e9
        rollout_time = str(round(rollout_time, 2))
        self.logger['delta_t'] = time.time_ns()
        process_time = (self.logger['delta_t'] - self.logger['batch_delta_t']) / 1e9
        process_time = str(round(process_time, 2))
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])

        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        lr = self.logger['lr']

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
        print(f"Rollout took: {rollout_time} secs", flush=True)
        print(f"Processing took: {process_time} secs", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

