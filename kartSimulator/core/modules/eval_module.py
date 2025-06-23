import sys

import gymnasium as gym
import torch

from kartSimulator.core.networks.standard_network import FFNetwork
from kartSimulator.core.networks.snn_network_small import SNN_small
from torch.distributions import Categorical
import numpy as np
import kartSimulator.core.snn_utils as SNN_utils

class Evaluator:

    def __init__(self, env, actor_state, NType):

        self.env = env

        self.actor_model = actor_state

        print(f"Testing {self.actor_model}", flush=True)

        # If the actor model is not specified, then exit
        if self.actor_model == '':
            print(f"Didn't specify model file. Exiting.", flush=True)
            sys.exit(0)

        # Determine if the action space is continuous or discrete
        if isinstance(env.action_space, gym.spaces.Box):
            print("Using a continuous action space")
            self.continuous = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            print("Using a discrete action space")
            self.continuous = False
        else:
            raise NotImplementedError("The action space type is not supported.")

        # Extract out dimensions of observation and action spaces
        obs_dim = self.env.observation_space.shape[0]

        if self.continuous:
            self.act_dim = self.env.action_space.shape[0]
        else:
            act_dim = self.env.action_space.n

        # Build our policy the same way we build our actor model in PPO
        # policy = ActorNetwork(obs_dim, act_dim)
        if NType == "ANN":
            self.actor = FFNetwork(obs_dim, act_dim)
        else:
            self.actor = SNN_small(obs_dim, act_dim, num_steps=32, add_weight=0.1)

        # Load in the actor model saved by the PPO algorithm
        self.actor.load_state_dict(torch.load(self.actor_model, weights_only=False))

    def eval(self, n_eval_episodes=5):


        for episode in range(n_eval_episodes):
            while True:

                obs = self.env.reset(options={})[0]

                terminated = False
                truncated = False

                # number of timesteps so far
                t = 0

                # Logging data
                ep_len = 0  # episodic length
                ep_ret = 0  # episodic return

                while not terminated and not truncated:
                    t += 1

                    # Query deterministic action from policy
                    if not self.continuous:
                        action_probs = self.actor(obs)
                        action = torch.argmax(action_probs, dim=-1).item()
                    else:
                        action = self.actor(obs).detach().numpy()

                    obs, rew, terminated, truncated, _ = self.env.step(action)

                    # Sum all episodic rewards as we go along
                    ep_ret += rew

                # Track episodic length
                ep_len = t
                print("reward for this episode :", ep_ret)

            mean_reward = np.mean(ep_ret)
            return mean_reward

    def eval_policy_ANN(self, n_eval_episodes=5):
        """
        Evaluates the given actor (policy) in the environment for a fixed number of episodes.

        :param n_eval_episodes: Number of episodes to evaluate over.
        :return: Mean reward over all episodes.
        """
        rewards = []

        for episode in range(n_eval_episodes):
            obs, _ = self.env.reset(options={})
            terminated = False
            truncated = False
            episode_reward = 0

            while not (terminated or truncated):
                # Get action from the actor (policy network)

                logits = self.actor.forward(obs)

                dist = Categorical(logits=logits)
                action = dist.sample().detach().numpy()

                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            print("reward for this episode :", episode_reward)

        # Calculate mean reward over all evaluation episodes
        mean_reward = np.mean(rewards)
        return mean_reward

    def eval_policy_SNN(self, n_eval_episodes=5, num_steps=32, threshold=None, shift=None):
        """
        Evaluates the given actor (policy) in the environment for a fixed number of episodes.

        :param n_eval_episodes: Number of episodes to evaluate over.
        :return: Mean reward over all episodes.
        """
        rewards = []

        for episode in range(n_eval_episodes):
            obs, _ = self.env.reset(options={})
            terminated = False
            truncated = False
            episode_reward = 0

            while not (terminated or truncated):
                # Get action from the actor (policy network)

                obs_st = SNN_utils.generate_spike_trains(obs,
                                                         num_steps=num_steps,
                                                         threshold=threshold,
                                                         shift=shift)

                logits, _ = self.actor.forward(obs_st)  # Assuming 'forward' method in actor handles the action logic

                dist = Categorical(logits=logits)
                action = dist.sample().detach().numpy()

                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            print("reward for this episode :", episode_reward)

        # Calculate mean reward over all evaluation episodes
        mean_reward = np.mean(rewards)
        return mean_reward


    def eval_sb3(self, kind, deterministic):
        self.baselines.eval(self.env, kind, deterministic)
