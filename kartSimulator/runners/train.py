
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
            'lr': 0.00017887220926944984,
            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'num_minibatches': 80,
            'gae_lambda': 0.9642298634023644,
            'verbose': 2,
            'num_steps': 32,
            'add_weight': 0.06,
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
            'lr': 0.00174,
            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'num_minibatches': 112,
            'gae_lambda': 0.9665,
            'verbose': 2,
        }

        hyperparameters_gazebo_12 = {
            'timesteps_per_batch': 2560,
            'gamma': 0.9723727953756666,
            'ent_coef': 0.0013258186155261317,
            'n_updates_per_iteration': 9,
            'lr': 0.0009756929138687892,
            'num_minibatches': 128,
            'gae_lambda': 0.9445920005319369,

            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'verbose': 2,
        }

        hyperparameters_gazebo_12_SNN = {
            'timesteps_per_batch': 2560,
            'gamma': 0.9723727953756666,
            'ent_coef': 0.00013258186155261317,
            'n_updates_per_iteration': 9,
            'lr': 0.009756929138687892,
            'num_minibatches': 128,
            'gae_lambda': 0.9445920005319369,

            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'verbose': 2,
            'num_steps': 100,
        }

        hyperparameters_gazebo_28 = {
            'timesteps_per_batch': 4096,
            'gamma': 0.9660523586032805,
            'ent_coef': 0.001004986805501314,
            'n_updates_per_iteration': 5,
            'lr': 0.0013550942974094524,
            'num_minibatches': 112,
            'gae_lambda': 0.9247541731348968,

            'clip': 0.2,
            'max_grad_norm': 0.5,
            'render_every_i': 10,
            'target_kl': None,
            'verbose': 2,
        }

        seed = 66676

        saving = {
            "ghost": True,
            "models": True,
            "wandb": True,
        }

        network_type = "SNN"

        # create agent : kart

        trainer.set_hyperparameters(hyperparameters_gazebo_12_SNN)
        trainer.set_saving(saving)
        trainer.set_seed(seed)
        trainer.set_network(network_type)

        # env = gym.make('LunarLander-v2')
        env = env_factory.createEnv(track_type, track_name, None)

        total_timesteps = 1_200_000
        #actor_state = "saves/projects/real-turtle/mantis-1/ppo_actor.pth"
        #critic_state = "saves/projects/real-turtle/mantis-1/ppo_critic.pth"

        actor_state = None
        critic_state = None

        trainer.train(env, total_timesteps, actor_state, critic_state)





