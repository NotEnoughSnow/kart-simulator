import numpy as np
import h5py
import pygame
import kartSimulator.sim.replay_simple_env as replay_simple_env


class ReplayGhosts:

    def __init__(self, location, episode=None):
        total_ghost_ep, total_ghost_ts, info = self.read_ghost_file(location)

        num_episodes = info["num_eps"]

        kwargs = {}

        mode = "all"

        if mode == "consecutive":
            num_agent = 1
        if mode == "all":
            num_agent = num_episodes

        env = replay_simple_env.KartSim(num_agents=num_agent, **kwargs)
        self.launch(env, total_ghost_ep, total_ghost_ts, mode)

    def launch(self, env, episodes, episode_lengths, mode="all"):
        if mode == "all":
            self.launch_all(env, episodes, episode_lengths)
        if mode == "consecutive":
            self.launch_con(env, episodes, episode_lengths)

    def read_ghost_file(self, location):
        # Loading data from HDF5
        with h5py.File(location, 'r') as f:
            # Read metadata
            env_name = f.attrs['env_name']
            track_name = f.attrs['track_name']
            total_num_ep = f.attrs['total_num_ep']

            # Initialize lists for episodes and their lengths
            total_ghost_ep = []
            total_ghost_ts = []

            # Read episodes data
            for i in range(total_num_ep):
                grp = f[f'episode_{i + 1}']
                ep_length = grp.attrs['episode_length']
                actions = grp['actions'][:]

                total_ghost_ep.append(actions)
                total_ghost_ts.append(ep_length)

        print("Data loaded successfully!")
        # print("Environment Name:", env_name)
        # print("Track Name:", track_name)
        # print("Total Number of Episodes:", total_num_ep)
        # print("Episode Lengths:", total_ghost_ts)
        # print("First Episode Actions Shape:", total_ghost_ep[0].shape)

        info = {
            "num_eps": total_num_ep,
            "track_name": track_name,
            "env_name": env_name,
        }

        return total_ghost_ep, total_ghost_ts, info

    def launch_all(self, env, episodes, episode_lengths):
        running = True

        max_ep_len = max(episode_lengths)

        max_ep_timesteps = max(episode_lengths)
        num_episodes = len(episodes)
        num_actions = episodes[0].shape[1]

        padded_episodes = np.zeros((num_episodes, max_ep_timesteps, num_actions))

        for i, episode in enumerate(episodes):
            padded_episodes[i, :len(episode), :] = episode

        new_episodes = padded_episodes.transpose(1, 0, 2)

        print(np.shape(new_episodes))

        while True:

            env.reset()
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            expert_data = []
            expert_episode = []

            for i in range(max_ep_len):
                array_of_positions = new_episodes[i]

                # for event in pygame.event.get():
                #    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                #        env.reset()

                obs, reward, terminated, truncated, _ = env.step(array_of_positions)

                # print(obs)

                total_reward += reward
                steps += 1

        # env.close()

    def launch_con(self, env, episodes, episode_lengths):
        running = True

        max_ep_len = max(episode_lengths)

        total_reward = 0.0
        steps = 0

        expert_data = []
        expert_episode = []

        max_ep_timesteps = max(episode_lengths)
        num_episodes = len(episodes)
        num_actions = episodes[0].shape[1]

        for i in range(num_episodes):

            env.reset()

            episode = episodes[i]
            print(f"playing episode {i} for {len(episodes[i])} timesteps")

            for j in range(0, len(episode), 1):
                action = episode[j]

                obs, reward, terminated, truncated, _ = env.step([action])

                # print(obs)

                total_reward += reward
                steps += 1

        print(steps)

        # env.close()
