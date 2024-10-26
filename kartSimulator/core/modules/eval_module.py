import sys

import gymnasium as gym
import torch

from kartSimulator.core.networks.standard_network import FFNetwork


class Evaluator:


    def __init__(self, env, actor_state):

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
        self.actor = FFNetwork(obs_dim, act_dim)

        # Load in the actor model saved by the PPO algorithm
        self.actor.load_state_dict(torch.load(self.actor_model))

    def eval(self):


        # Rollout until user kills process
        while True:
            obs = self.env.reset()[0]

            terminated = False
            truncated = False

            # number of timesteps so far
            t = 0

            # Logging data
            ep_len = 0  # episodic length
            ep_ret = 0  # episodic return

            while not terminated and not truncated:
                t += 1

                # Query deterministic action from policy and run it
                #action, _ = policy(obs)

                #env.render()

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

            # returns episodic length and return in this iteration
            return ep_len, ep_ret


    def eval_sb3(self, kind, deterministic):
        self.baselines.eval(self.env, kind, deterministic)
