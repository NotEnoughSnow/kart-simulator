import os

import h5py
import numpy as np
import pygame


class Launcher():

    def __init__(self,
                 env,
                 record,
                 save_dir,
                 player_name,
                 expert_ep_count,
                 expert_steps_count):
        self.env = env

        self.record = record
        self.save_dir = save_dir
        self.player_name = player_name
        self.expert_ep_count = False if expert_ep_count is None else expert_ep_count
        self.steps_needed = False if expert_steps_count is None else expert_steps_count

    def launch(self, env):
        running = True
        expert_run = []
        expert_ep_pos = []
        i = 0

        info = {
            "player_name": self.player_name,
            "num_episodes": self.expert_ep_count,
        }

        total_steps = 0

        while running and ( i < self.expert_ep_count or total_steps < self.steps_needed) :
            _, reset_info = env.reset(options={})
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            player_moved = False  # Reset flag at the start of each episode

            expert_episode = []
            # print("---------------------------------------------")

            while not terminated and not truncated:
                action = 0

                keys = pygame.key.get_pressed()

                if env.metadata["name"] == "kart2D grid_env":
                    if keys[pygame.K_w]:
                        action = 1
                    if keys[pygame.K_s]:
                        action = 2
                    if keys[pygame.K_a]:
                        action = 3
                    if keys[pygame.K_d]:
                        action = 4
                if env.metadata["name"] == "kart2D steer_env":
                    if keys[pygame.K_w]:
                        action = 1
                    if keys[pygame.K_SPACE]:
                        action = 2
                    if keys[pygame.K_a]:
                        action = 3
                    if keys[pygame.K_d]:
                        action = 4
                if env.metadata["name"] == "kart2D steer_gazebo":
                    if keys[pygame.K_w]:
                        action = 1
                    if keys[pygame.K_a]:
                        action = 2
                    if keys[pygame.K_d]:
                        action = 3

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        env.reset()

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Check if the player has moved
                if not player_moved and not action == 0:
                    player_moved = True

                # print(obs)

                # Append timestep data only if player has moved
                if player_moved:
                    expert_episode.append([steps, obs, action, reward, terminated, truncated])
                    steps += 1

                    total_steps += 1

            if truncated:
                print("hit a wall, ")
                print(f"total rewards this ep: {total_reward}")
                pass

            if terminated:
                print("finished, ")
                print(f"total rewards this ep: {total_reward}")
                # TODO times
                pass

            print("timesteps so far ", total_steps)

            # wrap expert data and steps in expert episode
            expert_ep_pos.append(reset_info["player pos"])

            # build expert run
            expert_run.append(expert_episode)
            if self.record:
                i += 1

        if self.record:

            base_dir = self.save_dir + "expert_data\\"

            run_number = 1
            while True:
                run_path = os.path.join(base_dir, f'ExpertData_{self.player_name}_{run_number}.hdf5')
                if not os.path.exists(run_path):
                    break
                # os.makedirs(run_path)
                run_number += 1

            print("saving expert data to ", run_path)

            # write expert runs to file then exit application
            self.write_file(expert_run, expert_ep_pos, info, run_path)

            print(" numepisodes :", len(expert_run))
            print(" num timesteps for the first episode :", len(expert_run[0]))
            print(" data of the first timestep :", expert_run[0][0])

            print(" ep lens :", expert_ep_pos)

            env.close()
            exit()

        env.close()
    def write_file(self, expert_run, expert_ep_lens, info, filename):
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