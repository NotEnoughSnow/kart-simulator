import random

import numpy as np
import h5py
import pygame
import kartSimulator.sim.replay_simple_env as replay_simple_env

class ReplayGhosts:

    """
    def __init__(self, location, episode=None):

        batches, batch_lengths, _, info = self.read_ghost_file(location)

        num_episodes = info["num_batches"]

        # all : replay all trained agents at the same time
        # con : replay agents 1 by 1
        # batch : replay trained agents batch by batch
        mode = "batch"

        self.launch(batches, batch_lengths, info, mode=mode)
    """

    def __init__(self, locations):


        mode = "batch"

        if mode == "all":
            self.replay_all_mul(locations)
        if mode == "batch":
            self.replay_batch_mul(locations)

    """
    def launch(self, batches, batch_lengths, info, mode="all"):
        if mode == "all":
            self.launch_all(batches, batch_lengths, info)
        if mode == "con":
            self.launch_con(batches, batch_lengths, info)
        if mode == "batch":
            self.launch_batch(batches, batch_lengths, info)
    """

    def read_ghost_file(self, location):
        # Loading data from HDF5
        with h5py.File(location, 'r') as f:
            # Load metadata
            env_name = f.attrs['env_name']
            track_name = f.attrs['track_name']
            total_num_batches = f.attrs['total_num_batches']
            max_ep_len = f.attrs['max_ep_len']

            # Prepare lists to hold the loaded data
            batches = []
            batch_episode_lens = []
            batch_lens = []


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
                batch_episode_lens.append(episode_lengths)
                batch_lens.append(len(episode_lengths))

        info = {
            "env_name": env_name,
            "track_name": track_name,

            "num_batches": total_num_batches,
            "num_episodes": sum(batch_lens),
            # FIXME this shouldnt be static, it needs to be taken from the max_ep_len parameter
            "max_num_ep_len": max_ep_len,

            "action_len": len(batches[0][0][0]),
        }


        print("Data loaded successfully!")
        print("Environment Name:", info["env_name"])
        print("Track Name:", info["track_name"])
        print("Total Number of Batches:", info["num_batches"])
        print("Max number of episode length:", info["max_num_ep_len"])
        print("Number of episodes per batch:", batch_lens)
        print("First Batch Episode lengths:", batch_episode_lens[0])
        print("Action len:", info["action_len"])
        print("Total number of episodes:", info["num_episodes"])


        print("+++++++++++++++++++++++++++++++++++++++++++++")


        return batches, batch_episode_lens, batch_lens, info


    def load_and_process_hdf5_files(self, file_paths):
        all_batches = []
        all_batch_episode_lens = []
        all_batch_lens = []

        g_info = {"total_num_batches": 0,
                  "total_num_episodes": 0,
                  "max_ep_len": [],
                  "num_files": len(file_paths),
                  "colors": [],
                  }
        colors = []

        all_info = []

        for i, file_path in enumerate(file_paths):
            print("Loading file: ", file_path)

            batches, batch_episode_lens, batch_lens, info = self.read_ghost_file(file_path)
            all_batches.append(batches)
            all_batch_episode_lens.append(batch_episode_lens)
            all_batch_lens.append(batch_lens)

            g_info["max_ep_len"].append(info["max_num_ep_len"])
            g_info["total_num_batches"] += info["num_batches"]
            g_info["total_num_episodes"] += info["num_episodes"]
            all_info.append(info)

        available_colors = [(255, 0, 0, 255), (0, 0, 255, 255), (0, 255, 0, 255), (255, 0, 255, 255)]

        colors = random.sample(available_colors, len(file_paths))

        g_info["max_ep_len"] = max(g_info["max_ep_len"])
        g_info["colors"] = colors

        all_info.insert(0, g_info)

        print("processed all files")
        print("total num batches: ", g_info["total_num_batches"])
        print("total number of episodes: ", g_info["total_num_episodes"])
        print("max length of an episode (all trainings): ", g_info["max_ep_len"])
        print("number of trainings: ", g_info["num_files"])
        print("colors: ", g_info["colors"])

        print(all_info[0])

        return all_batches, all_batch_episode_lens, all_batch_lens, all_info

    def process_batches(self, batches, batch_episode_lens, batch_lens, all_info):
        combined_batches = []
        combined_batch_episode_lens = []

        all_episodes = []
        all_episode_lengths = []


        # Combine all batches and lengths
        for batch, batch_length in zip(batches, batch_episode_lens):
            combined_batches.extend(batch)
            combined_batch_episode_lens.extend(batch_length)

        print(f"num batches in both trainings: {len(combined_batches)}")


        # Combine all batches and lengths
        for batch, batch_length in zip(combined_batches, combined_batch_episode_lens):
            all_episodes.extend(batch)
            all_episode_lengths.extend(batch_length)

        print(f"num episodes in both: {len(all_episodes)}")


        # TODO repalce, can get from file
        max_ep_len = max(all_episode_lengths)

        all_info[0]["max_ep_len"] = max(all_episode_lengths)

        #num_episodes = len(all_episodes)

        # Create a padded array for all episodes
        padded_episodes = np.zeros((all_info[0]["total_num_episodes"], all_info[0]["max_ep_len"], 2))

        for i, episode in enumerate(all_episodes):
            padded_episodes[i, :len(episode), :] = episode

        # Reshape the array for timestep-wise iteration
        new_episodes = padded_episodes.transpose(1, 0, 2)


        return new_episodes


    def replay_all_mul(self, file_paths):
        batches, batch_episode_lens, batch_lens, all_info = self.load_and_process_hdf5_files(file_paths)

        new_episodes = self.process_batches(batches, batch_episode_lens, batch_lens, all_info)

        num_episodes = [info["num_episodes"] for info in all_info[1:]]

        print("launching")

        kwargs = {}
        env = replay_simple_env.KartSim(num_agents=num_episodes, colors=all_info[0]["colors"], **kwargs)

        running = True
        while running:
            env.reset()

            for i in range(all_info[0]["max_ep_len"]):
                array_of_positions = new_episodes[i]
                done = env.step(array_of_positions, 1, 1)


    def process_batches_batch(self, all_batches, batch_episode_lens, batch_lens, all_info):
        combined_batches = []
        combined_max_ep_lens = []

        num_trainings = len(all_batches)
        max_num_batches = max(len(training_batches) for training_batches in all_batches)

        # Initialize array for padded batches
        max_ep_len = all_info[0]["max_ep_len"]
        total_num_agents = sum(max(len(batch) for batch in training_batches) for training_batches in all_batches)
        padded_batches_shape = (max_num_batches, max_ep_len, total_num_agents, 2)
        padded_batches = np.zeros(padded_batches_shape)

        agent_idx_offset = 0

        for training_idx, training_batches in enumerate(all_batches):
            max_batches_in_training = len(training_batches)
            max_agents_in_training = max(len(batch) for batch in training_batches)

            for batch_idx, batch in enumerate(training_batches):
                num_episodes_in_batch = len(batch)
                max_ep_len_in_batch = max(batch_episode_lens[training_idx][batch_idx])

                combined_max_ep_lens.append(max_ep_len_in_batch)

                for episode_idx, episode in enumerate(batch):
                    padded_batches[batch_idx, :len(episode), agent_idx_offset + episode_idx, :] = episode

            agent_idx_offset += max_agents_in_training

        return padded_batches, combined_max_ep_lens

    def replay_batch_mul(self, file_paths):
        all_batches, batch_episode_lens, batch_lens, all_info = self.load_and_process_hdf5_files(file_paths)

        new_batches, max_num_batches = self.process_batches_batch(all_batches, batch_episode_lens, batch_lens, all_info)

        print("Processing complete")

        max_ep_len = all_info[0]["max_ep_len"]
        num_trainings = all_info[0]["num_files"]

        num_agents_per_training = [max(len(batch) for batch in training_batches) for training_batches in all_batches]
        colors = all_info[0]["colors"]

        print("yah yeet")
        print(colors)
        print(num_agents_per_training)


        kwargs = {}
        env = replay_simple_env.KartSim(num_agents=num_agents_per_training, colors=colors, **kwargs)

        running = True

        while running:

            # Iterate over the maximum number of batches
            for batch_idx in range(new_batches.shape[0]):
                # Get the current batch
                current_batch = new_batches[batch_idx]

                env.reset()

                # Iterate over the timesteps within the current batch
                for timestep in range(max_ep_len):
                    array_of_positions = current_batch[timestep]
                    done = env.step(array_of_positions, batch_idx, new_batches.shape[0])

                    if done:
                        break



    # not currently used

    def launch_all(self, batches, batch_lengths, batch_lens, info):

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

        print("say what :", np.shape(batches))

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

        print("say what :", np.shape(all_padded_episodes))
        print("huh :", all_max_ep_len)


        print(info)

        # Initialize your environment
        kwargs = {}
        env = replay_simple_env.KartSim(num_agents=[10], colors=[(255,0,0,255)], **kwargs)

        running = True


        while running:
            k = 0

            for new_episodes, max_ep_len in zip(all_padded_episodes, all_max_ep_len):

                print("for batch :", k)

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

                k = k + 1

        # env.close()