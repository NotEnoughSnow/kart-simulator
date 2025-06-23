import os
import sys

import h5py
import torch
import yaml

#from kartSimulator.core import baselines
from kartSimulator.core.ppo_IM import PPO_IM
from kartSimulator.core.ppo import PPO
from kartSimulator.core.ppo_snn import PPO_SNN
from kartSimulator.sim import utils


class Trainer():

    def __init__(self, save_config, expert_file_name):

        self.expert_data = None

        if expert_file_name is not None:
            self.expert_data = self.load_expert_data(expert_file_name=expert_file_name)

        self.hyperparameters = {
            'timesteps_per_batch': 4024,
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
            'verbose': 2,
        }

        # ANN, SNN
        self.Ntype = "ANN"

        # TODO should be outside
        self.save_dir = save_config["save_dir"]
        self.project_name = save_config["project_name"]
        self.run_name = save_config["run_name"]
        self.description = save_config["description"]

        self.saving = {
            "ghost": False,
            "models": False,
            "wandb": False,
        }

    def set_saving(self, saving):
        self.saving = saving

    def set_seed(self, seed):
        self.hyperparameters["seed"] = seed

    def set_network(self, Ntype):
        self.Ntype = Ntype

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def train(self,
              env,
              total_timesteps,
              actor_state,
              critic_state,
              ):

        base_dir = self.save_dir + f"{self.project_name}/"

        if (self.saving["wandb"] or self.saving["ghost"] or self.saving["models"]) is True:

            save_path, ver_number = utils.get_next_run_directory_mod(base_dir, self.run_name)


            train_config = self.save_train_data(env,
                                                save_path,
                                                ver_number,
                                                self.Ntype,
                                                total_timesteps,
                                                self.hyperparameters,
                                                self.description)

        else:
            train_config = None
            save_path = None
            ver_number = ""

        if self.Ntype == "ANN":
            result_actor_state, result_critic_state = self.train_ANN(env=env,
                                                       total_timesteps=total_timesteps,
                                                       save_path=save_path,
                                                       run_name=f"{self.run_name}-{ver_number}",
                                                       train_config=train_config,
                                                       actor_model=actor_state,
                                                       critic_model=critic_state)
        if self.Ntype == "SNN":
            result_actor_state, result_critic_state = self.train_SNN(env,
                                                       total_timesteps,
                                                       save_path,
                                                       train_config,
                                                       actor_state,
                                                       critic_state)

        return result_actor_state, result_critic_state

    def train_ANN(self,
                  env,
                  total_timesteps,
                  save_path,
                  run_name,
                  train_config,
                  actor_model,
                  critic_model,
                  ):
        # TODO change this back
        model = PPO(env=env,
                    save_model=self.saving["models"],
                    record_ghost=self.saving["ghost"],
                    record_output=False,
                    save_dir=save_path,
                    description=self.description,
                    run_name = run_name,
                    record_wandb=self.saving["wandb"],
                    train_config=train_config,
                    expert_data=self.expert_data ,
                    project_name = self.project_name,
                    **self.hyperparameters)

        if actor_model != None and critic_model != None:
            print(f"Loading in {actor_model} and {critic_model}...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.critic.load_state_dict(torch.load(critic_model))
            print(f"Successfully loaded.", flush=True)
        elif actor_model != None or critic_model != None:  # Don't train from scratch if user accidentally forgets actor/critic model
            print(
                f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
            sys.exit(0)
        else:
            print(f"Training from scratch.", flush=True)

        model.learn(total_timesteps=total_timesteps)

        _, actor_state = model.get_actor()
        _, critic_state = model.get_actor()

        return actor_state, critic_state

    def train_SNN(self,
                  env,
                  total_timesteps,
                  save_path,
                  train_config,
                  actor_model,
                  critic_model,
                  ):

        model = PPO_SNN(env=env,
                        save_model=self.saving["models"],
                        record_ghost=self.saving["ghost"],
                        record_output=False,
                        save_dir=save_path,
                        record_wandb=self.saving["wandb"],
                        train_config=train_config,
                        expert_data=self.expert_data,
                        project_name = self.project_name,
                        **self.hyperparameters)

        if actor_model != None and critic_model != None:
            print(f"Loading in {actor_model} and {critic_model}...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.critic.load_state_dict(torch.load(critic_model))
            print(f"Successfully loaded.", flush=True)
        elif actor_model != None or critic_model != None:  # Don't train from scratch if user accidentally forgets actor/critic model
            print(
                f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
            sys.exit(0)
        else:
            print(f"Training from scratch.", flush=True)

        model.learn(total_timesteps=total_timesteps)

        _, actor_state = model.get_actor()
        _, critic_state = model.get_actor()

        return actor_state, critic_state

    def train_sb3(self,
                  env,
                  total_timesteps,
                  save_path,
                  train_config,
                  ):
        pass
        #baselines.train(env, self.save_dir, self.record_output, self.experiment_name, steps=total_timesteps)

    def save_train_data(self, env, save_dir, ver_number, alg, total_timesteps, hyperparameters, description):
        # env name
        # map
        # obs
        # hyperparameters

        parameters = {
            "project": self.project_name,
            "name": self.run_name,
            "version": ver_number,
            "algorithm": alg,
            "env name": env.metadata.get("name", "None"),
            "evn track": env.metadata.get("track", "None"),
            "obs types": env.metadata.get("obs_seq", "None"),
            "total timesteps": total_timesteps,
            "hyperparameters": hyperparameters,
            "description": description,
        }

        save_dir = save_dir
        yaml_file_path = save_dir + "/parameters.yaml"

        if not os.path.exists(yaml_file_path):
            with open(yaml_file_path, 'w') as file:
                yaml.dump(parameters, file, default_flow_style=False)

            print(f"Parameters saved to {yaml_file_path}")

        return parameters

    def load_expert_data(self, expert_file_name):
        """
        = [ R1, R2, R2, ..]
        R = [ E_T1, E_T2, E_T3, ..]
        E_T = [time, obs, actions, terminated, truncated]

        :param expert_file_name:
        :return:
        """

        expert_run = []
        expert_ep_lens = []
        expert_info = {}

        with h5py.File(expert_file_name, "r") as f:
            # Load metadata
            for key, value in f.attrs.items():
                expert_info[key] = value

            # There's only one expert run
            run_group = f["expert_run"]

            for episode_key in run_group.keys():
                episode_group = run_group[episode_key]
                total_steps = episode_group.attrs['total_steps']
                episode_data = []

                for timestep_key in episode_group.keys():
                    timestep_group = episode_group[timestep_key]
                    time = timestep_group['time'][()]
                    observations = timestep_group['observations'][()]
                    actions = timestep_group['actions'][()]
                    reward = timestep_group['reward'][()]
                    terminated = timestep_group['terminated'][()]
                    truncated = timestep_group['truncated'][()]
                    episode_data.append([time, observations, actions, reward, terminated, truncated])

                expert_run.append(episode_data)
                expert_ep_lens.append(total_steps)

        print("successfully loaded expert data")

        return expert_run, expert_ep_lens, expert_info
