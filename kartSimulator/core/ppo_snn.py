import time
import sys

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Categorical
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
import kartSimulator.core.snn_utils as SNN_utils

import h5py

import wandb
from joblib import Parallel, delayed
import pickle

from kartSimulator.core.actor_network import ActorNetwork
from kartSimulator.core.critic_network import CriticNetwork
from kartSimulator.core.snn_network import SNN
from kartSimulator.core.snn_network_small import SNN_small


class PPO_SNN:

    def __init__(self, env, record_ghost, save_model, record_output, save_dir, record_wandb, train_config, **hyperparameters):

        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)

        # Determine if the action space is continuous or discrete
        if isinstance(env.action_space, gym.spaces.Box):
            print("Using a continuous action space")
            self.continuous = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            print("Using a discrete action space")
            self.continuous = False
        else:
            raise NotImplementedError("The action space type is not supported.")

        print(train_config)

        self.record_wandb = record_wandb
        self.record_output = record_output
        self.record_ghost = record_ghost
        self.save_model = save_model

        if self.record_wandb:
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="PPO-SNN-Lunar-Landing",

                # track hyperparameters and run metadata
                config=train_config
            )

        self.run_directory = save_dir

        if self.record_output:
            # Specify the file where you want to save the output
            output_file = f"{save_dir}/graph_data.txt"
            self.output_file = open(output_file, 'w')
            sys.stdout = Tee(sys.stdout, self.output_file)

        if (record_ghost or record_output or save_model) is True:
            print(f"Saving to '{self.run_directory}' after training")


        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]

        if self.continuous:
            self.act_dim = env.action_space.shape[0]
        else:
            self.act_dim = env.action_space.n

        # TODO automate
        self.threshold = torch.tensor([1.5, 1.5, 5, 5, 3.14, 5, 1, 1])
        self.shift = np.array([1.5, 1.5, 5, 5, 3.14, 5, 0, 0])

        print(f"obs shape :{self.act_dim} \n"
              f"action shape :{self.act_dim}")

        # Initialize actor and critic networks
        # self.actor = ActorNetwork(self.obs_dim, self.act_dim)
        # self.critic = CriticNetwork(self.obs_dim, 1)

        print(self.num_steps)

        self.actor = SNN_small(self.obs_dim, self.act_dim, self.num_steps)
        self.critic = SNN_small(self.obs_dim, 1, self.num_steps)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix for continuous action spaces
        if self.continuous:
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

    def parallel_rollout(self):

        results = Parallel(n_jobs=4)(delayed(self.rollout)() for _ in range(4))

        # Initialize empty lists/tensors for each variable
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []
        batch_ghosts = []

        # Combine results from all processes
        for result in results:
            batch_obs.append(result[0])  # Collect tensors
            batch_acts.append(result[1])  # Collect tensors
            batch_log_probs.append(result[2])  # Collect 1D tensors
            batch_rews.extend(result[3])  # Collect 2D lists
            batch_lens.extend(result[4])  # Collect 1D lists
            batch_vals.extend(result[5])  # Collect 2D lists
            batch_dones.extend(result[6])  # Collect 2D lists
            batch_ghosts.extend(result[7])  # Collect 2D vectors

        # Concatenate tensors along the first dimension
        batch_obs = torch.cat(batch_obs, dim=0)
        batch_acts = torch.cat(batch_acts, dim=0)
        batch_log_probs = torch.cat(batch_log_probs, dim=0)

        # Now you have combined tensors/lists for batch_obs, batch_acts, etc.
        # print("Combined batch_obs:", batch_obs)
        # print("Combined batch_acts:", batch_acts)
        # print("Combined batch_log_probs:", batch_log_probs)
        # print("Combined batch_rews:", batch_rews)
        # print("Combined batch_lens:", batch_lens)
        # print("Combined batch_vals:", batch_vals)
        # print("Combined batch_dones:", batch_dones)

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_ghosts

    def learn(self, total_timesteps):

        print(
            f"Learning... Running {self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps",
            end='')

        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        total_ghost_ep = []
        total_ghost_ts = []

        while t_so_far < total_timesteps:

            parallel = False

            if parallel:
                batch_obs, batch_obs_st, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_ghosts = self.parallel_rollout()
            else:
                batch_obs, batch_obs_st, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_ghosts = self.rollout()

            #batch_obs, batch_obs_st, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_ghosts = self.rollout()
            # batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_ghosts = self.rollout()

            self.logger['batch_delta_t'] = time.time_ns()

            total_ghost_ep.append(batch_ghosts)
            total_ghost_ts.append(batch_lens)

            # Calculate advantage at k-th iteration
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)

            # TODO entry
            V_st, _ = self.critic(batch_obs_st)
            if self.decode_type == "first":
                V = SNN_utils.decode_first_spike_batched(V_st).squeeze()
            if self.decode_type == "count":
                V = SNN_utils.get_spike_counts_batched(V_st).squeeze()
            if self.decode_type == "lrl":
                V = V_st.squeeze()

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

            if self.record_wandb:
                wandb.log({
                    "train/explained_variance": explained_variance,
                }, step=self.logger['t_so_far'])

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
                    mini_obs_st = batch_obs_st[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t)
                    V, curr_log_probs, dist, entropy_loss = self.evaluate(mini_obs_st, mini_acts)

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

                    # Before the backpropagation step
                    initial_weights = self.actor.fc1.weight.clone().detach()
                    initial_biases = self.actor.fc1.bias.clone().detach()

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

                    updated_weights = self.actor.fc1.weight.clone().detach()
                    updated_biases = self.actor.fc1.bias.clone().detach()

                    # Calculate changes (L2 norm, or other metrics)
                    weight_change = torch.norm(updated_weights - initial_weights, p=2)
                    bias_change = torch.norm(updated_biases - initial_biases, p=2)

                    if self.record_wandb:
                        wandb.log({"snn/weight_change_fc1": weight_change,
                                   "snn/bias_change_fc1": bias_change}, step=self.logger['t_so_far'])

                    # calculating metrics
                    total_loss = actor_loss + critic_loss
                    policy_gradient_loss = actor_loss
                    value_loss = critic_loss

                    loss_arr.append(actor_loss.detach())

                    if self.record_wandb:
                        wandb.log({
                            "train/clip_range": self.clip,
                            "train/clip_fraction": clip_fraction,
                            "train/approx_kl": approx_kl,
                            "train/policy_gradient_loss": policy_gradient_loss,
                            "train/value_loss": value_loss,
                            "train/loss": total_loss,
                        }, step=self.logger['t_so_far'])

                # Approximating KL Divergence
                if self.target_kl is not None and approx_kl > self.target_kl:
                    print("target kl reached, stopping early")
                    break

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

        # TODO inherit from game
        if self.record_ghost:
            print("Saving ghost data")
            self.save_ghost(self.env,
                            batches=total_ghost_ep,
                            batch_lengths=total_ghost_ts, )

        print("Finished successfully!")

        if self.record_wandb:
            wandb.finish()

        self.output_file.close()

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
        batch_obs_st = []
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

                obs_st = SNN_utils.generate_spike_trains(obs,
                                                         num_steps=self.num_steps,
                                                         threshold=self.threshold,
                                                         shift=self.shift)
                batch_obs_st.append(obs_st)
                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs_st)

                # TODO entry
                val_st, _ = self.critic(obs_st)
                if self.decode_type == "first":
                    val = SNN_utils.decode_first_spike(val_st)
                if self.decode_type == "count":
                    val = SNN_utils.get_spike_counts(val_st)
                if self.decode_type == "lrl":
                    val = val_st

                obs, rew, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated

                # print("fps", info["fps"])
                # print("position", info["position"])

                ghost_ep.append(info.get("position", [0, 0]))

                # TODO simplify batch actions and ghost data
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_vals.append(val.flatten())

                # print(rew)

                # TODO add fps

                ep_t += 1

            batch_ghosts.append(ghost_ep)

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

            ep_rew_mean = np.sum(ep_rews)
            # print(ep_rew_mean)

            if self.record_wandb:
                wandb.log({
                    "rollout/ep_rew_mean": ep_rew_mean,
                    "rollout/ep_len_mean": (ep_t + 1),
                }, step=self.logger['t_so_far'])

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_obs_st = torch.tensor(np.array(batch_obs_st), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        # batch_rtgs = self.compute_rtgs(batch_rews)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_obs_st, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones, batch_ghosts

    def get_action(self, obs_st):
        """
        Queries an action from the actor network, should be called from rollout.

        Parameters:
            obs - the observation at the current timestep

        Return:
            action - the action to take, as a numpy array
            log_prob - the log probability of the selected action in the distribution
        """

        spk_output, spikes = self.actor(obs_st)

        avg_spike_time, spike_ratio = SNN_utils.compute_spike_metrics(spikes)

        if self.continuous:
            # For continuous action spaces
            # TODO entry
            if self.decode_type == "first":
                mean = SNN_utils.decode_first_spike(spk_output)
            if self.decode_type == "count":
                mean = SNN_utils.get_spike_counts(spk_output)
            if self.decode_type == "lrl":
                mean = spk_output

            dist = MultivariateNormal(mean, self.cov_mat)
        else:
            # For discrete action spaces
            # TODO entry
            if self.decode_type == "first":
                logits = SNN_utils.decode_first_spike(spk_output)
            if self.decode_type == "count":
                logits = SNN_utils.get_spike_counts(spk_output)
            if self.decode_type == "lrl":
                logits = spk_output

            dist = Categorical(logits=logits)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        if self.record_wandb:
            wandb.log({
                "snn/avg_spike_time": avg_spike_time.item(),
                "snn/spike_ratio": spike_ratio.item()}, step=self.logger['t_so_far'])

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs_st, batch_acts):
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

        # Query critic network for a value V for each batch_obs
        # TODO entry
        V_st, _ = self.critic(batch_obs_st)
        if self.decode_type == "first":
            V = SNN_utils.decode_first_spike_batched(V_st).squeeze()
        if self.decode_type == "count":
            V = SNN_utils.get_spike_counts_batched(V_st).squeeze()
        if self.decode_type == "lrl":
            V = V_st.squeeze()

        spk_output, _ = self.actor(batch_obs_st)

        # Calculate the log probabilities of batch actions using most recent actor network
        if self.continuous:
            # TODO entry
            if self.decode_type == "first":
                mean = SNN_utils.decode_first_spike_batched(spk_output)
            if self.decode_type == "count":
                mean = SNN_utils.get_spike_counts_batched(spk_output)
            if self.decode_type == "lrl":
                mean = spk_output

            dist = MultivariateNormal(mean, self.cov_mat)
        else:
            # TODO entry
            if self.decode_type == "first":
                logits = SNN_utils.decode_first_spike_batched(spk_output)
            if self.decode_type == "count":
                logits = SNN_utils.get_spike_counts_batched(spk_output)
            if self.decode_type == "lrl":
                logits = spk_output

            dist = Categorical(logits=logits)


        # Calculate entropy loss for regularization
        entropy_loss = -dist.entropy().mean()

        if self.record_wandb:
            wandb.log({
                "train/entropy_loss": entropy_loss,
            }, step=self.logger['t_so_far'])

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
        self.decode_type = "lrl"
        self.num_steps = 50

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

    def save_ghost(self, env, batches, batch_lengths):
        # Saving data to HDF5
        with h5py.File(f'{self.run_directory}/ghost.hdf5', 'w') as f:
            # Add metadata
            f.attrs['env_name'] = env.metadata.get("name", "None")
            f.attrs['track_name'] = env.metadata.get("track", "None")
            f.attrs['total_num_batches'] = len(batches)
            f.attrs['max_ep_len'] = env.metadata.get("reset_time", 0) + 2

            # Create group for batches
            batches_group = f.create_group('batches')

            for i, (batch, batch_length) in enumerate(zip(batches, batch_lengths)):
                batch_group = batches_group.create_group(f'batch_{i + 1}')
                batch_group.attrs['batch_length'] = batch_length

                # Create group for episodes within the batch
                episodes_group = batch_group.create_group('episodes')

                for j, (ep_data, ep_length) in enumerate(zip(batch, batch_length)):
                    ep_group = episodes_group.create_group(f'episode_{j + 1}')
                    ep_group.attrs['episode_length'] = ep_length

                    # Create dataset for actions within the episode
                    ep_group.create_dataset('actions', data=ep_data)


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure the output is written immediately

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                # Ignore the error if the file is already closed
                pass

    """
    def compute_rtgs(self, batch_rews):
    '''
        Compute the Reward-To-Go of each timestep in a batch given the rewards.

        Parameters:
            batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Return:
            batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
    '''
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
    """
