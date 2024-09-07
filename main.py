import gymnasium as gym
import yaml
import os
import sys

import torch
import csv
import ast
import numpy as np
import h5py

from kartSimulator.core.arguments import get_args
from kartSimulator.core import eval_policy
from kartSimulator.core.ppo import PPO
from kartSimulator.core.ppo_gym import PPO_gym
from kartSimulator.core.ppo_snn import PPO_SNN
from kartSimulator.core.ppo_one_iter import PPO as PPO_ONE
from kartSimulator.core.line_policy import L_Policy
import kartSimulator.evolutionary.core as EO
import kartSimulator.core.make_graphs as graph_generator
import kartSimulator.sim.observation_types as obs_types

from kartSimulator.core.replay_ghosts import ReplayGhosts

from kartSimulator.core.actor_network import ActorNetwork
from kartSimulator.core.standard_network import FFNetwork

import kartSimulator.core.baselines as baselines

import kartSimulator.sim.sim2D as kart_sim
import kartSimulator.sim.empty as empty_sim
import kartSimulator.sim.base_env as base_env
import kartSimulator.sim.simple_env as simple_env

import kartSimulator.sim.utils as utils

import kartSimulator.sim_turtlebot.sim_turtlebot as turtlebot_sim
import kartSimulator.sim_turtlebot.calibrate as turtlebot_calibrate
import pygame


def play(env, record, save_dir, player_name="Amin", expert_ep_count=3):
    running = True
    expert_run = []
    expert_ep_lens = []
    i = 0

    info = {
        "player_name": player_name,
        "num_episodes": expert_ep_count,
    }

    while running and i < expert_ep_count:
        env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        player_moved = False  # Reset flag at the start of each episode

        expert_episode = []

        while not terminated and not truncated:
            action = torch.zeros(2)

            keys = pygame.key.get_pressed()

            if env.metadata["name"] == "kart2D simple_env":

                if keys[pygame.K_w]:
                    action[0] = -1
                if keys[pygame.K_s]:
                    action[0] = +1
                if keys[pygame.K_d]:
                    action[1] = +1
                if keys[pygame.K_a]:
                    action[1] = -1
            else:
                if keys[pygame.K_w]:
                    action[1] = 1
                if keys[pygame.K_SPACE]:
                    action[1] = -1
                if keys[pygame.K_d]:
                    action[0] = 1
                if keys[pygame.K_a]:
                    action[0] = -1

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Check if the player has moved
            if not player_moved and not np.all(action.numpy() == 0):
                player_moved = True

            # print(obs)

            # Append timestep data only if player has moved
            if player_moved:
                expert_episode.append([steps, obs, action.numpy(), reward, terminated, truncated])
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
        expert_ep_lens.append(steps)

        # build expert run
        expert_run.append(expert_episode)
        if record:
            i += 1

    if record:

        base_dir = save_dir + "expert_data/"

        run_number = 1
        while True:
            run_path = os.path.join(base_dir, f'ExpertData_{player_name}_{run_number}.hdf5')
            if not os.path.exists(run_path):
                break
            # os.makedirs(run_path)
            run_number += 1

        print("saving expert data to ", run_path)

        # write expert runs to file then exit application
        write_file(expert_run, expert_ep_lens, info, run_path)

        print(" num episodes :", len(expert_run))
        print(" num timesteps for the first episode :", len(expert_run[0]))
        print(" data of the first timestep :", expert_run[0][0])

        print(" ep lens :", expert_ep_lens)

        env.close()
        exit()

    env.close()


def write_file(expert_run, expert_ep_lens, info, filename):
    with h5py.File(filename, "w") as f:
        # Save metadata as general attributes
        for key, value in info.items():
            f.attrs[key] = value

        # Create a group for the single expert run
        run_group = f.create_group("expert_run")

        # Iterate through episodes and their corresponding lengths
        for episode_index, (episode_data, episode_length) in enumerate(zip(expert_run, expert_ep_lens)):
            episode_group = run_group.create_group(f"episode_{episode_index}")
            episode_group.attrs['total_steps'] = episode_length

            # Iterate through timesteps and save their data with zero-padded keys
            for timestep_index, timestep in enumerate(episode_data):
                timestep_key = f"timestep_{timestep_index:04d}"  # Zero-padded index
                timestep_group = episode_group.create_group(timestep_key)
                timestep_group.create_dataset("time", data=timestep[0])
                timestep_group.create_dataset("observations", data=np.array(timestep[1], dtype=float))
                timestep_group.create_dataset("actions", data=np.array(timestep[2], dtype=float))
                timestep_group.create_dataset("reward", data=timestep[3])
                timestep_group.create_dataset("terminated", data=timestep[4])
                timestep_group.create_dataset("truncated", data=timestep[5])


def save_train_data(env, save_dir, experiment_name, ver_number, alg, total_timesteps, hyperparameters):
    # env name
    # map
    # obs
    # hyperparameters

    parameters = {
        "type": experiment_name,
        "version": ver_number,
        "algorithm": alg,
        "env name": env.metadata.get("name", "None"),
        "evn track": env.metadata.get("track", "None"),
        "obs types": env.metadata.get("obs_seq", "None"),
        "total timesteps": total_timesteps,
        "hyperparameters": hyperparameters,
    }

    save_dir = save_dir + f"{experiment_name}/"
    yaml_file_path = save_dir + "parameters.yaml"

    if not os.path.exists(yaml_file_path):
        with open(yaml_file_path, 'w') as file:
            yaml.dump(parameters, file, default_flow_style=False)

        print(f"Parameters saved to {yaml_file_path}")

    return parameters


def train(env,
          total_timesteps,
          alg,
          experiment_name,
          save_dir,
          record_tb,
          record_ghost,
          save_model,
          record_wandb,
          iteration_type,
          hyperparameters,
          ):
    actor_model = ''
    critic_model = ''

    base_dir = save_dir + f"{alg}/"

    save_path, ver_number = utils.get_next_run_directory(base_dir, experiment_name)

    train_config = save_train_data(env, base_dir, experiment_name, ver_number, alg, total_timesteps, hyperparameters)

    if alg == "default":

        model = PPO(env=env,
                    save_model=save_model,
                    record_ghost=record_ghost,
                    record_tb=record_tb,
                    save_dir=save_path,
                    record_wandb=record_wandb,
                    train_config=train_config,
                    **hyperparameters)

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

        if iteration_type == "one":
            model = PPO_ONE(env=env, **hyperparameters)
            model.learn(total_timesteps=1)
        else:
            model.learn(total_timesteps=total_timesteps)
    if alg == "snn":

        model = PPO_SNN(env=env,
                        save_model=save_model,
                        record_ghost=record_ghost,
                        record_tb=record_tb,
                        save_dir=save_path,
                        record_wandb=record_wandb,
                        train_config=train_config,
                        **hyperparameters)

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

        if iteration_type == "one":
            model = PPO_ONE(env=env, **hyperparameters)
            model.learn(total_timesteps=1)
        else:
            model.learn(total_timesteps=total_timesteps)
    if alg == "baselines":
        # TODO fix save_dir not needing experiment name

        baselines.train(env, save_dir, record_tb, experiment_name, steps=total_timesteps)


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

        # Determine if the action space is continuous or discrete
        if isinstance(env.action_space, gym.spaces.Box):
            print("Using a continuous action space")
            continuous = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            print("Using a discrete action space")
            continuous = False
        else:
            raise NotImplementedError("The action space type is not supported.")

        # Extract out dimensions of observation and action spaces
        obs_dim = env.observation_space.shape[0]
        if continuous:
            act_dim = env.action_space.shape[0]
        else:
            act_dim = env.action_space.n

        # Build our policy the same way we build our actor model in PPO
        # policy = ActorNetwork(obs_dim, act_dim)
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


def replay(replay_dir, mode):
    # replay = ReplayGhosts(replay_dir, replay_ep)
    replay = ReplayGhosts(replay_dir, mode)


def make_graphs(graph_file):
    # Extract absolute file paths
    graph_generator.graph_data(graph_file)


def optimize():
    # TODO
    EO.run()


def main(args):
    # timesteps_per_batch : n_steps
    # num_minibatches : batch_size
    # n_updates_per_iteration : n_epochs
    # ent_coef : ent_coef
    # gamma : gamma
    # gae-lambda : gae_lambda
    # render_every_i : stats_window_size

    hyperparameters = {
        'timesteps_per_batch': 1024,
        'max_timesteps_per_episode': 700,
        'gamma': 0.999,
        'ent_coef': 0.01,
        'n_updates_per_iteration': 4,
        'lr': 0.0004,
        'clip': 0.2,
        'max_grad_norm': 0.5,
        'render_every_i': 10,
        'target_kl': None,
        'num_minibatches': 64,
        'gae_lambda': 0.98,
    }

    # environment selection
    # simple_env has free movement
    # base_env has car like movement
    env_fn = simple_env

    # list of observations:
    # DISTANCE : distance to goal
    # TARGET_ANGLE : angle to target
    # POSITION : general position
    # ROTATION : general angles (not available for simple_env)
    # VELOCITY : single value speed of the agent
    # LIDAR : vision rays
    # LIDAR_CONV : vision rays with conv1d
    obs = [obs_types.DISTANCE,
           obs_types.TARGET_ANGLE,
           ]

    # keyword arguments for the environment
    # reset_time : num timesteps after which the episode will terminate
    # track_type :
    #   boxes - a simple S_curve, loaded with a map loader
    #   generator - generates a simple corner around the player, changes every reset
    #   simple_goal - randomly or statically spawns a random goal with no obstacles, changes every reset
    # player args :
    #   rad_velocity - ...
    #   0.1 rad/frames, 2pi in 1.281 sec
    #   example : 0.0379 rad/frames, for frames = 48
    #   or        1.82 rad/sec
    #
    #   max_velocity - ...
    #   1 meter in 4.546 sec or 0.22 meters in 1 sec
    #   at 0.22 m/s, the bot should walk 1 meter in 4.546 seconds
    #   MAX_VELOCITY = 4 * 0.22 * PPM

    track_args = {
        "boxes_file": "boxes.txt",
        "sectors_file": "sectors_box.txt",

        "corridor_size": 50,

        "spawn_range": 400,
        "fixed_goal": [200, -200],

        "initial_pos": [300, 450]
    }

    simple_env_player_args = {
        "player_acc_rate": 15,
        "max_velocity": 4,
        "bot_size": 0.192,
        "bot_weight": 1,
    }
    base_env_player_args = {
        "player_acc_rate": 6,
        "player_break_rate": 2,
        "max_velocity": 2,
        "rad_velocity": 2 * 2.84,
        "bot_size": 0.192,
        "bot_weight": 1,
    }


    env_args = {
        "obs_seq": obs,
        "reset_time": 2000,
        "track_type": "boxes",
        "track_args": track_args,
        "player_args": simple_env_player_args if env_fn == simple_env else base_env_player_args,
    }

    # Parameters for training
    # total_timesteps : total number of training timesteps
    # record_tensorboard : to record tensorboard data
    # record_ghost : to save ghost data (files for replays)
    # save_model : to save actor and critic models
    # iteration_type : mul for default mode, one to run a single iteration
    # alg : default, baselines, snn
    train_parameters = {
        "total_timesteps": 300000,
        "record_tb": False,
        "record_ghost": False,
        "save_model": False,
        "record_wandb": False,
        "iteration_type": "mul",
        "alg": "default",
    }

    # Save parameters
    # experiment_name : change to test out different conditions
    experiment_name = "LL4"
    save_dir = "./saves/"

    # Parameters for testing
    # deterministic : deterministic evaluation value (for stable baselines)
    deterministic = False

    # Parameters for imitation learning
    # record_expert_data : to record data for imitation learning
    # expert_ep_count : number of episodes to record
    record_expert_data = False
    expert_ep_count = 1
    player_name = "Amin"

    # Parameters for replays
    replay_files = ["saves/default/B2/ver_1/ghost.hdf5"]
    mode = "batch"

    # parameters for making graphs
    graph_file = "saves/default/LL3/ver_1/graph_data.txt"

    # Load from YAML
    # with open(yaml_file_path, 'r') as file:
    #    loaded_parameters_yaml = yaml.load(file, Loader=yaml.FullLoader)

    # print(loaded_parameters_yaml)

    if args.mode == "play":
        env = env_fn.KartSim(render_mode="human", train=False, **env_args)
        play(env=env, save_dir=save_dir, player_name=player_name, record=record_expert_data,
             expert_ep_count=expert_ep_count)

    if args.mode == "train":
        env = gym.make('LunarLander-v2')
        # env = env_fn.KartSim(render_mode=None, train=True, **env_args)
        train(env=env,
              **train_parameters,
              experiment_name=experiment_name,
              save_dir=save_dir,
              hyperparameters=hyperparameters,
              )

    if args.mode == "test":
        # env = env_fn.KartSim(render_mode="human", train=False, **env_args)
        env = gym.make('LunarLander-v2', render_mode="human")

        test(env,
             alg=train_parameters["alg"],
             type=experiment_name,
             deterministic=deterministic,
             actor_model=f'./saves/default/{experiment_name}/ver_1/ppo_actor.pth'),

    if args.mode == "replay":
        replay(replay_dir=replay_files,
               mode=mode,
               )

    if args.mode == "graph":
        make_graphs(graph_file=graph_file)

    if args.mode == "optimize":
        # TODO implement evolutionary optimization
        pass


if __name__ == "__main__":
    args = get_args()

    # you can also directly set the args
    # args.mode = "train"
    # modes : play, train, test, graph, replay

    args.mode = "play"

    main(args)
