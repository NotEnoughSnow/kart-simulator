from kartSimulator.core.modules.train_module import Trainer
from kartSimulator.core.modules.curr_module import CurriculumTrainer

from kartSimulator.sim.maps.track_factory import TrackFactory


class Train_curr():

    def __init__(self, track_type, env_factory, save_config):

        save_config["save_dir"] = save_config["save_dir"] + "projects/"

        # yeet setup
        trainer = Trainer(save_config)


        seeds = [9482, 1234, 5678, 5830, 129103, 1219312]


        hyperparameters = {
            'timesteps_per_batch': 4096,
            'gamma': 0.9804294200427209,
            'ent_coef': 0.0022855955790885233,
            'n_updates_per_iteration': 11,
            # multiply by 10 for ANN
            'lr': 0.0006789186367493657,
            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': 0.5,
            'num_minibatches': 80,
            'gae_lambda': 0.955885656511228,
            'seed': 928,
            'verbose': 2,
        }

        saving = {
            "ghost": True,
            "models": True,
            "wandb": True,
        }

        network_type = "ANN"

        trainer.set_hyperparameters(hyperparameters)
        trainer.set_saving(saving)
        trainer.set_network(network_type)

        # yeet end setup

        # yeet train 1

        # Adjust rewards
        rew_adj = {
            "passive": 0,
            "dist": 0,
            "act_dist": 1,
            "sector_time": 0.7,
            "steer": 0.2,
        }

        env_factory.set_rew_adj(rew_adj)

        total_timesteps = 50000

        env = env_factory.createEnv(track_type, "map_1", None)

        trainer.set_seed(seeds[0])


        trainer.train(env, total_timesteps, None, None)

        # yeet train 2

        # Adjust rewards
        rew_adj = {
            "passive": 0,
            "dist": 0,
            "act_dist": 0.5,
            "sector_time": 0,
            "steer": 1,
        }

        env_factory.set_rew_adj(rew_adj)

        total_timesteps = 100000

        env = env_factory.createEnv(track_type, "map_2", None)

        trainer.set_seed(seeds[1])


        trainer.train(env=env,
                      total_timesteps=total_timesteps,
                      actor_state="saves/projects/curriculum-project/zaza-1/ppo_actor.pth",
                      critic_state="saves/projects/curriculum-project/zaza-1/ppo_critic.pth",
                      )



        # yeet train 3

        #saving["wandb"] = True

        # Adjust rewards
        rew_adj = {
            "passive": 0,
            "dist": 0,
            "act_dist": 0.5,
            "sector_time": 0.2,
            "steer": 0.6,
        }

        env_factory.set_rew_adj(rew_adj)

        total_timesteps = 100000

        env = env_factory.createEnv(track_type, "map_3", None)

        trainer.set_seed(seeds[2])


        trainer.train(env=env,
                      total_timesteps=total_timesteps,
                      actor_state="saves/projects/curriculum-project/zaza-2/ppo_actor.pth",
                      critic_state="saves/projects/curriculum-project/zaza-2/ppo_critic.pth",
                      )

        # yeet train 4

        #saving["wandb"] = True

        # Adjust rewards
        rew_adj = {
            "passive": 0,
            "dist": 0,
            "act_dist": 0.4,
            "sector_time": 0.5,
            "steer": 0.7,
        }

        env_factory.set_rew_adj(rew_adj)

        total_timesteps = 100000

        env = env_factory.createEnv(track_type, "map_4", None)

        trainer.set_seed(seeds[3])


        trainer.train(env=env,
                      total_timesteps=total_timesteps,
                      actor_state="saves/projects/curriculum-project/zaza-3/ppo_actor.pth",
                      critic_state="saves/projects/curriculum-project/zaza-3/ppo_critic.pth",
                      )

        # yeet train 5

        # saving["wandb"] = True


        # Adjust rewards
        rew_adj = {
            "passive": 0,
            "dist": 0,
            "act_dist": 0.7,
            "sector_time": 0.3,
            "steer": 0.7,
        }

        env_factory.set_rew_adj(rew_adj)

        total_timesteps = 300000

        env = env_factory.createEnv(track_type, "boxes", None)

        trainer.set_seed(seeds[4])


        trainer.train(env=env,
                      total_timesteps=total_timesteps,
                      actor_state="saves/projects/curriculum-project/zaza-4/ppo_actor.pth",
                      critic_state="saves/projects/curriculum-project/zaza-4/ppo_critic.pth",
                      )

        # yeet train 6

        # Adjust rewards
        rew_adj = {
            "passive": 0,
            "dist": 0,
            "act_dist": 0.8,
            "sector_time": 1,
            "steer": 0.2,
        }

        env_factory.set_rew_adj(rew_adj)

        total_timesteps = 300000

        env = env_factory.createEnv(track_type, "boxes", None)

        trainer.set_seed(seeds[5])


        trainer.train(env=env,
                      total_timesteps=total_timesteps,
                      actor_state="saves/projects/curriculum-project/zaza-5/ppo_actor.pth",
                      critic_state="saves/projects/curriculum-project/zaza-5/ppo_critic.pth",
                      )



