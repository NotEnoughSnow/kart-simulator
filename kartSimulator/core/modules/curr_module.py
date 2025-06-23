from kartSimulator.core.modules.train_module import Trainer
from kartSimulator.sim.maps.track_factory import TrackFactory


class CurriculumTrainer:

    def __init__(self, env_name, env_args, track_args, curr_tracks_info, save_config):
        self.env_name = env_name
        self.env_args = env_args
        self.curr_tracks_info = curr_tracks_info
        self.trainer = Trainer(save_config)
        self.actor_state = None
        self.critic_state = None

        self.track_args = track_args


        self.default_hyperparameters = {
            'timesteps_per_batch': 4096,
            'gamma': 0.9634703441998751,
            'ent_coef': 0.004797586864549939,
            'n_updates_per_iteration': 7,
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

    def set_trainer(self, hyperparameters=None, saving=None, network_type="ANN", seed=1234):
        hyperparameters = hyperparameters or self.default_hyperparameters
        saving = saving or {"ghost": False, "models": False, "wandb": False}

        self.trainer.set_hyperparameters(hyperparameters)
        self.trainer.set_saving(saving)
        self.trainer.set_seed(seed)
        self.trainer.set_network(network_type)

    def train_on_env(self, total_timesteps, env_args, curr_map, actor_state, critic_state):

        # Update track_args with only the necessary fields from curr_tracks_info
        self.track_args.update({
            "boxes_file": self.curr_tracks_info[curr_map]["boxes_file"],
            "sectors_file": self.curr_tracks_info[curr_map]["sectors_file"],
            "initial_pos": self.curr_tracks_info[curr_map]["initial_pos"],
            "rand goal": self.curr_tracks_info[curr_map]["rand goal"],
        })

        self.env_args = env_args

        # Create the track
        track = TrackFactory.create_track("curr", **self.track_args)
        self.env_args["track"] = track

        # Create environment and train
        env = self.env_name.KartSim(render_mode=None, train=True, **self.env_args)

        self.trainer.train(env, total_timesteps, actor_state, critic_state)


