
from kartSimulator.core.modules.train_module import Trainer



class Train():

    def __init__(self, track_type, track_name, env_factory, save_config):

        save_config["save_dir"] = save_config["save_dir"] + "projects\\"

        expert_file_name = ".\\saves\\expert_data\\ExpertData_Amin_1.hdf5"

        trainer = Trainer(save_config=save_config, expert_file_name=expert_file_name)


        hyperparameters = {
            'timesteps_per_batch': 4096,
            'gamma': 0.9634703441998751,
            'ent_coef': 0.004797586864549939,
            'n_updates_per_iteration': 7,
            # multiply by 10 for ANN
            'lr': 0.00017887220926944984,
            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': 0.5,
            'num_minibatches': 80,
            'gae_lambda': 0.9642298634023644,
            'seed': 928,
            'verbose': 2,
        }

        seed = 1234

        saving = {
            "ghost": False,
            "models": False,
            "wandb": False,
        }

        network_type = "ANN"

        # create agent : kart

        trainer.set_hyperparameters(hyperparameters)
        trainer.set_saving(saving)
        trainer.set_seed(seed)
        trainer.set_network(network_type)

        # env = gym.make('LunarLander-v2')
        env = env_factory.createEnv(track_type, track_name, None)

        total_timesteps = 10000

        trainer.train(env, total_timesteps, None, None)





