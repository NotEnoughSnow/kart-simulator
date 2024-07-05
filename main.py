import sys

import torch
import csv
import ast
import numpy as np
import h5py

from kartSimulator.core.arguments import get_args
from kartSimulator.core import eval_policy
from kartSimulator.core.ppo import PPO
from kartSimulator.core.ppo_snn import SNNPPO
from kartSimulator.core.ppo_one_iter import PPO as PPO_ONE
from kartSimulator.core.line_policy import L_Policy
import kartSimulator.evolutionary.core as EO
import kartSimulator.sim.observation_types as obs_types

from kartSimulator.core.actor_network import ActorNetwork
from kartSimulator.core.standard_network import FFNetwork

import kartSimulator.core.baselines as baselines

import kartSimulator.sim.sim2D as kart_sim
import kartSimulator.sim.empty as empty_sim
import kartSimulator.sim.base_env as base_env
import kartSimulator.sim.simple_env as simple_env

import kartSimulator.sim_turtlebot.sim_turtlebot as turtlebot_sim
import kartSimulator.sim_turtlebot.calibrate as turtlebot_calibrate
import pygame


def play(env, record, player_name="Amin", expert_ep_count=3):
    running = True
    expert_run = []
    i = 0

    while running and i < expert_ep_count:
        env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        expert_data = []
        expert_episode = []

        while not terminated and not truncated:
            action = torch.zeros(5)

            keys = pygame.key.get_pressed()
            # key controls
            # if keys[pygame.K_w]:
            #    action[3] = 1
            # if keys[pygame.K_SPACE]:
            #    action[4] = 1
            # if keys[pygame.K_d]:
            #    action[1] = 1
            # if keys[pygame.K_a]:
            #    action[2] = 1

            if keys[pygame.K_w]:
                action[0] = -1
            if keys[pygame.K_s]:
                action[0] = +1
            if keys[pygame.K_d]:
                action[1] = +1
            if keys[pygame.K_a]:
                action[1] = -1

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()

            obs, reward, terminated, truncated, _ = env.step(action)

            # print(obs)

            # build expert episode
            expert_data.append([steps, obs, action.numpy(), reward, terminated, truncated])

            total_reward += reward
            steps += 1

        if truncated:
            print("hit a wall")
            print(f"total rewards this ep:{total_reward}")

        if terminated:
            print("finished")
            print(f"total rewards this ep:{total_reward}")
            # TODO times

        print(steps)

        # wrap expert data and steps in expert episode
        expert_episode = [expert_data, steps]

        # build expert run
        expert_run.append(expert_episode)
        if record:
            i += 1

    if record:
        # write expert runs to file then exit application
        write_file(expert_run, player_name)
        env.close()
        exit()

    env.close()


def write_file(expert_runs, player_name, filename="expert.hdf5"):
    with h5py.File(filename, "w") as f:
        for run_index, expert_episode in enumerate(expert_runs):
            run_group = f.create_group(f"run_{run_index}")
            # Store the player's name as an attribute of the run
            run_group.attrs['player_name'] = player_name

            # Access the elements directly by index
            episode_data = expert_episode[0]
            episode_steps = expert_episode[1]

            episode_group = run_group.create_group(f"episode_{run_index}")
            episode_group.attrs['total_steps'] = episode_steps

            # Create datasets for timestep data
            for timestep_index, timestep in enumerate(episode_data):
                timestep_group = episode_group.create_group(f"timestep_{timestep_index}")
                timestep_group.create_dataset("time", data=timestep[0])
                timestep_group.create_dataset("observations", data=np.array(timestep[1], dtype=float))
                timestep_group.create_dataset("actions", data=np.array(timestep[2], dtype=float))
                timestep_group.create_dataset("reward", data=timestep[3])
                timestep_group.create_dataset("terminated", data=timestep[4])
                timestep_group.create_dataset("truncated", data=timestep[5])


def train(env, total_timesteps, alg, type, logs_dir, record_tb, iterations, hyperparameters, actor_model, critic_model):
    if alg == "default":
        model = PPO(env=env, location=type, **hyperparameters)

        # Tries to load in an existing actor/critic model to continue training on
        if actor_model != '' and critic_model != '':
            print(f"Loading in {actor_model} and {critic_model}...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.critic.load_state_dict(torch.load(critic_model))
            print(f"Successfully loaded.", flush=True)
        elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
            print(
                f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
            sys.exit(0)
        else:
            print(f"Training from scratch.", flush=True)

        if iterations == "one":
            model = PPO_ONE(env=env, **hyperparameters)
            model.learn(total_timesteps=1)
        else:
            model.learn(total_timesteps=total_timesteps)
    if alg == "baselines":
        baselines.train(env, logs_dir, record_tb, type, steps=total_timesteps)


def test(env, alg, type, deterministic, actor_model):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    if alg == "default":
        print(f"Testing {actor_model}", flush=True)

        # If the actor model is not specified, then exit
        if actor_model == '':
            print(f"Didn't specify model file. Exiting.", flush=True)
            sys.exit(0)

        # Extract out dimensions of observation and action spaces
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Build our policy the same way we build our actor model in PPO
        #policy = ActorNetwork(obs_dim, act_dim)
        policy = FFNetwork(obs_dim, act_dim)

        # Load in the actor model saved by the PPO algorithm
        policy.load_state_dict(torch.load(actor_model))

        # Evaluate our policy with a separate module, eval_policy, to demonstrate
        # that once we are done training the model/policy with ppo.py, we no longer need
        # ppo.py since it only contains the training algorithm. The model/policy itself exists
        # independently as a binary file that can be loaded in with torch.

        eval_policy.eval_policy(policy=policy, env=env, render=True)

        # eval_policy.visualize(policy,env)

    if alg == "baselines":
        baselines.eval(env, type, deterministic)


def train_SNN(env, hyperparameters):
    model = SNNPPO(env=env, **hyperparameters)

    print(f"Training from scratch.", flush=True)

    model.learn(total_timesteps=100000)


def optimize():
    # TODO
    EO.run()


def main(args):
    hyperparameters = {
        'timesteps_per_batch': 256,
        'max_timesteps_per_episode': 2048,
        'gamma': 0.99,
        'ent_coef': 0.001,
        'n_updates_per_iteration': 10,
        'lr': 0.001,
        'clip': 0.2,
        'max_grad_norm': 0.5,
        'render_every_i': 10,
        'target_kl': None,
        'num_minibatches': 8,
    }

    # timesteps_per_batch : batch_size
    # max_timesteps_per_episode : n_steps
    # n_updates_per_iteration : n_epochs
    # render_every_i : stats_window_size

    env_fn = simple_env

    obs = [obs_types.DISTANCE,
           obs_types.TARGET_ANGLE, ]

    kwargs = {
        "obs_seq": obs,
    }

    type = "T3"
    logs_dir = "logs_300"
    deterministic = True
    total_timesteps = 300000

    if args.mode == "play":
        env = env_fn.KartSim(render_mode="human", train=False, **kwargs)
        play(env=env, record=False)

    if args.mode == "train":
        env = env_fn.KartSim(render_mode=None, train=True, **kwargs)
        train(env=env,
              total_timesteps=total_timesteps,
              alg=args.alg,
              type=type,
              logs_dir=logs_dir,
              record_tb=False,
              iterations="mul",
              hyperparameters=hyperparameters,
              actor_model=args.actor_model,
              critic_model=args.critic_model,
              )

    if args.mode == "test":
        env = env_fn.KartSim(render_mode="human", train=False, **kwargs)
        test(env,
             alg=args.alg,
             type=type,
             deterministic=deterministic,
             actor_model="ppo_actor.pth"),

    if args.mode == "snn":
        env = env_fn.KartSim(render_mode=None, train=True, **kwargs)
        train_SNN(env=env, hyperparameters=hyperparameters)

    if args.mode == "optimize":
        # TODO implement evolutionary optimization
        pass


if __name__ == "__main__":
    args = get_args()

    # you can also directly set the args
    # args.mode = "train"

    args.mode = "train"

    args.alg = "default"

    main(args)
