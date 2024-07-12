import numpy as np
import h5py
import pygame
import kartSimulator.sim.replay_simple_env as replay_simple_env

class ReplayGhosts:

    def __init__(self, location, episode=None):

        batches, batch_lengths, info = self.read_ghost_file(location)

        num_episodes = info["num_batches"]

        # all : replay all trained agents at the same time
        # con : replay agents 1 by 1
        # batch : replay trained agents batch by batch
        mode = "all"


        self.launch(batches, batch_lengths, info, mode=mode)

    def __init__(self, locations):
        self.replay_all_mul(locations)

    def launch(self, batches, batch_lengths, info, mode="all"):
        if mode == "all":
            self.launch_all(batches, batch_lengths, info)
        if mode == "con":
            self.launch_con(batches, batch_lengths, info)
        if mode == "batch":
            self.launch_batch(batches, batch_lengths, info)

    def read_ghost_file(self, location):
        # Loading data from HDF5
        with h5py.File(location, 'r') as f:
            # Load metadata
            env_name = f.attrs['env_name']
            track_name = f.attrs['track_name']
            total_num_batches = f.attrs['total_num_batches']

            # Prepare lists to hold the loaded data
            batches = []
            batch_lengths = []

            # Iterate through the batches
            for i in range(total_num_batches):
                batch_group = f[f'batches/batch_{i + 1}']
                batch_length = batch_group.attrs['batch_length']

                # Prepare lists to hold episodes within the batch
                episodes = []
                episode_lengths = []

                episodes_group = batch_group['episodes']
                for j in range(len(episodes_group)):
                    ep_group = episodes_group[f'episode_{j + 1}']
                    ep_length = ep_group.attrs['episode_length']
                    actions = np.array(ep_group['actions'])

                    episodes.append(actions)
                    episode_lengths.append(ep_length)

                batches.append(episodes)
                batch_lengths.append(episode_lengths)

        print("Data loaded successfully!")
        print("Environment Name:", env_name)
        print("Track Name:", track_name)
        print("Total Number of Batches:", total_num_batches)
        print("Number of episodes per batch:", len(batch_lengths[0]))
        print("First Batch Episode lengths:", batch_lengths[0])
        print("First Episode Actions Shape:", batches[0][0][0].shape)


        info = {
            "num_batches": total_num_batches,
            "num_ep": len(batch_lengths[0]),
            "track_name": track_name,
            "env_name": env_name,
        }

        return batches, batch_lengths, info

    def launch_all(self, batches, batch_lengths, info):

        # Combine all episodes and lengths from all batches
        all_episodes = []
        all_episode_lengths = []

        for batch, batch_length in zip(batches, batch_lengths):
            all_episodes.extend(batch)
            all_episode_lengths.extend(batch_length)

        max_ep_len = max(all_episode_lengths)

        num_episodes = len(all_episodes)
        num_actions = all_episodes[0].shape[1]

        # Create a padded array for all episodes
        padded_episodes = np.zeros((num_episodes, max_ep_len, num_actions))

        for i, episode in enumerate(all_episodes):
            padded_episodes[i, :len(episode), :] = episode

        # Reshape the array for timestep-wise iteration
        new_episodes = padded_episodes.transpose(1, 0, 2)

        kwargs = {}
        env = replay_simple_env.KartSim(num_agents=num_episodes, **kwargs)

        running = True

        while running:
            env.reset()
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            for i in range(max_ep_len):
                array_of_positions = new_episodes[i]

                # Process pygame events (if using pygame)
                # for event in pygame.event.get():
                #     if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                #         env.reset()

                obs, reward, terminated, truncated, _ = env.step(array_of_positions)

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            # You might want to break the outer while loop based on some condition
            # running = False  # For example, if you want to exit after one full playback

        # env.close()

    def launch_con(self, batches, batch_lengths, info):
        running = True

        kwargs = {}
        env = replay_simple_env.KartSim(num_agents=1, **kwargs)

        total_reward = 0.0
        steps = 0

        # Process batches
        all_padded_episodes = []
        all_max_ep_len = []

        for batch, batch_length in zip(batches, batch_lengths):
            max_ep_len = max(batch_length)
            num_episodes = len(batch)
            num_actions = batch[0].shape[1]

            for i in range(num_episodes):
                episode = batch[i]
                padded_episode = np.zeros((max_ep_len, num_actions))
                padded_episode[:len(episode), :] = episode
                all_padded_episodes.append(padded_episode)
                all_max_ep_len.append(len(episode))

        # Play episodes
        while running:
            for padded_episode, ep_len in zip(all_padded_episodes, all_max_ep_len):
                env.reset()
                print(f"Playing episode for {ep_len} timesteps")

                for j in range(ep_len):
                    action = padded_episode[j]

                    obs, reward, terminated, truncated, _ = env.step([action])

                    total_reward += reward
                    steps += 1

                    if terminated or truncated:
                        break

            # Exit the loop after playing all episodes
            running = False

        print(steps)
        # env.close()

    def launch_batch(self, batches, batch_lengths, info):
        all_padded_episodes = []
        all_max_ep_len = []

        for batch, batch_length in zip(batches, batch_lengths):
            max_ep_len = max(batch_length)
            num_episodes = len(batch)
            num_actions = batch[0].shape[1]

            # Create a padded array for the current batch
            padded_episodes = np.zeros((num_episodes, max_ep_len, num_actions))

            for i, episode in enumerate(batch):
                padded_episodes[i, :len(episode), :] = episode

            # Reshape the array for timestep-wise iteration
            new_episodes = padded_episodes.transpose(1, 0, 2)

            all_padded_episodes.append(new_episodes)
            all_max_ep_len.append(max_ep_len)

        # Initialize your environment
        kwargs = {}
        env = replay_simple_env.KartSim(num_agents=info["num_ep"], **kwargs)

        running = True

        while running:
            for new_episodes, max_ep_len in zip(all_padded_episodes, all_max_ep_len):
                env.reset()
                total_reward = 0.0
                steps = 0
                terminated = False
                truncated = False

                for i in range(max_ep_len):
                    array_of_positions = new_episodes[i]

                    # Process pygame events (if using pygame)
                    # for event in pygame.event.get():
                    #     if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    #         env.reset()

                    obs, reward, terminated, truncated, _ = env.step(array_of_positions)

                    total_reward += reward
                    steps += 1

                    if terminated or truncated:
                        break

                # You might want to break the outer loop based on some condition
                # running = False  # For example, if you want to exit after one full batch

        # env.close()


    def load_and_process_hdf5_files(self, file_paths):
        all_batches = []
        all_batch_lengths = []
        total_info = {"num_ep": 0}
        colors = []

        for i, file_path in enumerate(file_paths):
            print("Loading file: ", file_path)

            batches, batch_lengths, info = self.read_ghost_file(file_path)
            all_batches.append(batches)
            all_batch_lengths.append(batch_lengths)
            total_info["num_ep"] += info["num_ep"]
            colors.extend([(255, 0, 0,255), (0, 0, 255, 255), (0, 255, 0, 255), (255, 255, 0, 255)][i % 4] for _ in range(info["num_ep"]))

        print("processed all files")
        print("number of batches: ", len(all_batches))
        print("colors: ", len(colors))

        return all_batches, all_batch_lengths, total_info, colors


    def process_batches(self, batches, batch_lengths):
        all_episodes = []
        all_episode_lengths = []

        for batch, batch_length in zip(batches, batch_lengths):
            all_episodes.extend(batch)
            all_episode_lengths.extend(batch_length)

        max_ep_len = max(all_episode_lengths)
        num_episodes = len(all_episodes)
        num_actions = all_episodes[0].shape[1]

        padded_episodes = np.zeros((num_episodes, max_ep_len, num_actions))

        for i, episode in enumerate(all_episodes):
            padded_episodes[i, :len(episode), :] = episode

        new_episodes = padded_episodes.transpose(1, 0, 2)

        return new_episodes, max_ep_len, num_episodes

    def replay_batches(self, env, new_episodes, max_ep_len):
        running = True
        while running:
            env.reset()

            for i in range(max_ep_len):
                array_of_positions = new_episodes[i]
                env.step_mul(array_of_positions)


    def replay_all_mul(self, file_paths):
        batches, batch_lengths, info, colors = self.load_and_process_hdf5_files(file_paths)

        new_episodes, max_ep_len, num_episodes = self.process_batches(batches, batch_lengths)

        kwargs = {}
        env = replay_simple_env.KartSim(num_agents=num_episodes, colors=colors, **kwargs)

        self.replay_batches(env, new_episodes, max_ep_len)

