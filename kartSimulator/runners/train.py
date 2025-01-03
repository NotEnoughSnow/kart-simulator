
from kartSimulator.core.modules.train_module import Trainer



class Train():

    def __init__(self, track_type, track_name, env_factory, save_config):

        save_config["save_dir"] = save_config["save_dir"] + "projects/"

        #expert_file_name = "./saves/expert_data/ExpertData_Amin_1.hdf5"
        expert_file_name = None

        trainer = Trainer(save_config=save_config, expert_file_name=expert_file_name)


        hyperparameters_SMRE_SNN = {
            'timesteps_per_batch': 4096,
            'gamma': 0.9634703441998751,
            'ent_coef': 0.004797586864549939,
            'n_updates_per_iteration': 7,
            'lr': 0.0017887220926944984,
            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'num_minibatches': 80,
            'gae_lambda': 0.9642298634023644,
            'verbose': 2,
            'num_steps': 32,
            'add_weight': 0.02,
        }

        hyperparameters_GMRE_SNN = {
            'timesteps_per_batch': 3584,
            'gamma': 0.9717,
            'ent_coef': 0.00993,
            'n_updates_per_iteration': 9,
            'lr': 0.00174,
            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'num_minibatches': 112,
            'gae_lambda': 0.9665,
            'verbose': 2,
            'num_steps': 50,
            'add_weight': 0.05,
        }

        hyperparameters_SMRE_ANN = {
            'timesteps_per_batch': 4096,
            'gamma': 0.963,
            'ent_coef': 0.00479,
            'n_updates_per_iteration': 7,
            'lr': 0.000178,
            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'num_minibatches': 80,
            'gae_lambda': 0.9644,
            'verbose': 2,
        }

        hyperparameters_GMRE_ANN = {
            'timesteps_per_batch': 3584,
            'gamma': 0.9717,
            'ent_coef': 0.00993,
            'n_updates_per_iteration': 9,
            'lr': 0.000174,
            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'num_minibatches': 112,
            'gae_lambda': 0.9665,
            'verbose': 2,
        }

        seed = 485832

        saving = {
            "ghost": True,
            "models": True,
            "wandb": False,
        }

        network_type = "ANN"

        # create agent : kart

        trainer.set_hyperparameters(hyperparameters_GMRE_ANN)
        trainer.set_saving(saving)
        trainer.set_seed(seed)
        trainer.set_network(network_type)

        # env = gym.make('LunarLander-v2')
        env = env_factory.createEnv(track_type, track_name, None)

        total_timesteps = 300000

        #actor_state = "saves/projects/imitation-project/yeezy-1/ppo_actor.pth"
        #critic_state = "saves/projects/imitation-project/yeezy-1/ppo_critic.pth"

        actor_state = None
        critic_state = None

        trainer.train(env, total_timesteps, actor_state, critic_state)





