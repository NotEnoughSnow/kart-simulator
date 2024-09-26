import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import optuna

from kartSimulator.core.ppo import PPO
import numpy as np

from kartSimulator.sim.directional_env import KartSim
import kartSimulator.sim.observation_types as obs_types

from torch.distributions import Categorical

def eval_policy(actor, env, n_eval_episodes=5):
    """
    Evaluates the given actor (policy) in the environment for a fixed number of episodes.

    :param actor: The actor model from PPO_SNN (retrieved via model.get_actor()).
    :param env: The environment to evaluate on.
    :param n_eval_episodes: Number of episodes to evaluate over.
    :return: Mean reward over all episodes.
    """
    rewards = []

    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            # Get action from the actor (policy network)

            logits = actor.forward(obs)  # Assuming 'forward' method in actor handles the action logic


            dist = Categorical(logits=logits)
            action = dist.sample().detach().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    # Calculate mean reward over all evaluation episodes
    mean_reward = np.mean(rewards)
    return mean_reward


# Objective function for Optuna
def objective(trial):

    hyperparameters = {
        'timesteps_per_batch': trial.suggest_int('timesteps_per_batch', 512, 4096, step=512),
        'gamma': trial.suggest_float('gamma', 0.92, 0.9999, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.1, log=True),
        'n_updates_per_iteration': trial.suggest_int('n_updates_per_iteration', 1, 11, step=2),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'clip': 0.2,
        'max_grad_norm': 0.5,  # Fixed
        'render_every_i': 10,  # Fixed
        'target_kl': None,  # Fixed
        'num_minibatches': trial.suggest_int('num_minibatches', 16, 128, step=8),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
        'verbose': 1,  # Fixed
    }

    obs = [obs_types.LIDAR,
           obs_types.VELOCITY,
           obs_types.DISTANCE,
           obs_types.TARGET_ANGLE,
           ]

    track_args = {
        "boxes_file": "boxes.txt",
        "sectors_file": "sectors_box.txt",

        "corridor_size": 50,

        "spawn_range": 400,
        "fixed_goal": [200, -200],

        "initial_pos": [300, 450]
    }

    simple_env_player_args = {
        "player_acc_rate": 15,
        "max_velocity": 2,
        "bot_size": 0.192,
        "bot_weight": 1,
    }
    base_env_player_args = {
        "player_acc_rate": 1,
        "player_break_rate": 2,
        "max_velocity": 2,
        "rad_velocity": 2 * 2.84,
        "bot_size": 0.192,
        "bot_weight": 1,
    }

    env_args = {
        "obs_seq": obs,
        "reset_time": 2000,
        "track_type": "boxes",
        "track_args": track_args,
        #"player_args": simple_env_player_args if env_fn == simple_env else base_env_player_args,
        "player_args": simple_env_player_args,

    }

    env = KartSim(render_mode=None, train=False, **env_args)

    total_timesteps = 100000

    model = PPO(env=env,
                    save_model=False,
                    record_ghost=False,
                    record_output=False,
                    save_dir=False,
                    record_wandb=False,
                    train_config=None,
                    **hyperparameters)

    num_finishes, highest = model.learn(total_timesteps=total_timesteps)

    actor, actor_state_dict = model.get_actor()


    mean_reward = eval_policy(actor, env, n_eval_episodes=15)

    print("## results :")
    print("num_finishes :", num_finishes)
    print("highest :", highest)
    print("mean_reward :", mean_reward)

    result = num_finishes*2000 + highest*200 + mean_reward

    return result  # Metric for optuna


# Function to initialize the study and create the database
def initialize_study():
    study = optuna.create_study(direction="maximize", storage="sqlite:///./saves/hp_opt/optuna_hp.db",
                                study_name="racing_simple", load_if_exists=True)
    print("Database and study initialized!")


# Run the study in parallel processes
def run_study():
    study = optuna.create_study(direction="maximize", storage="sqlite:///./saves/hp_opt/optuna_hp.db",
                                study_name="racing_simple", load_if_exists=True)
    study.optimize(objective, n_trials=20)


if __name__ == "__main__":
    # Initialize the study (creates the database and tables)
    initialize_study()

    # Use multiprocessing to run parallel trials
    import multiprocessing

    #n_parallel_workers = multiprocessing.cpu_count()  # Adjust based on available resources
    n_parallel_workers = 2

    processes = []

    for _ in range(n_parallel_workers):
        p = multiprocessing.Process(target=run_study)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Load the study to retrieve the best result
    study = optuna.load_study(study_name="racing_simple", storage="sqlite:///./saves/hp_opt/optuna_hp.db")
    print("Best hyperparameters:", study.best_params)
