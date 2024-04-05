
import time

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim.adam import Adam


class L_Policy:


    def __init__(self, policy_class, env, **hyperparameters):
        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)
        assert (type(env.action_space) == gym.spaces.Box)

        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)  # ALG STEP 1


        # FIXME not SNN
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)


    def learn(self, total_timesteps):

        # get observation initially

        target_pos = torch.tensor([330.0, 110.0], dtype=torch.float32, requires_grad=True)

        ep_per_batch = 5
        num_epochs = 50

        for epoch in range(num_epochs):

            batch_loss = 0

            for _ in range(ep_per_batch):

                t = 0
                obs = self.env.reset()[0]

                # episode
                while t < 50:

                    action = self.actor(obs)

                    action = action.detach().numpy()


                    # get action from policy, requires observation
                    #action = self.line_policy(obs)

                    # take a timestep in env, requires action, returns observation
                    obs, _, _, _, _ = self.env.step(action)

                    current_position = torch.tensor([obs[0], obs[1]], dtype=torch.float32, requires_grad=True)

                    # actor loss
                    actor_loss = (current_position - target_pos) ** 2
                    batch_loss += actor_loss.sum()

                    t += 1

            batch_loss /= ep_per_batch

            self.actor_optim.zero_grad()
            batch_loss.backward(retain_graph=True)

            self.actor_optim.step()

            print(f"Epoch {epoch}, Loss: {batch_loss.item()}")


    def line_policy(self, obs):

        action = [0,0,0]

        return action


    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        self.lr = 0.005  # Learning rate of actor optimizer

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout







