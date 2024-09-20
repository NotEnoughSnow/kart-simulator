import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import optuna
import gymnasium as gym
from kartSimulator.core.ppo_snn import PPO_SNN
import numpy as np
import kartSimulator.core.snn_utils as SNN_utils
import torch

from torch.distributions import Categorical

def eval_policy(actor, env, n_eval_episodes=5, num_steps = 32, threshold = None, shift=None):
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

            obs_st = SNN_utils.generate_spike_trains(obs,
                                                     num_steps=num_steps,
                                                     threshold=threshold,
                                                     shift=shift)

            logits, _ = actor.forward(obs_st)  # Assuming 'forward' method in actor handles the action logic


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
        'timesteps_per_batch': trial.suggest_int('timesteps_per_batch', 512, 2048, step=512),
        'max_timesteps_per_episode': trial.suggest_int('max_timesteps_per_episode', 500, 1000),
        'gamma': trial.suggest_float('gamma', 0.95, 0.9999, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.1, log=True),
        'n_updates_per_iteration': trial.suggest_int('n_updates_per_iteration', 1, 10),
        'lr': trial.suggest_float('lr', 5e-4, 1e-2, log=True),
        'clip': trial.suggest_float('clip', 0.1, 0.3),
        'max_grad_norm': 0.5,  # Fixed
        'render_every_i': 10,  # Fixed
        'target_kl': None,  # Fixed
        'num_minibatches': trial.suggest_int('num_minibatches', 16, 128, step=16),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
        'verbose': 1,  # Fixed
    }

    # SNN-specific hyperparameters
    SNN_hyperparameters = {
        "num_steps": 32,  # Fixed
    }

    threshold = torch.tensor([1.5, 1.5, 5, 5, 3.14, 5, 1, 1])
    shift = np.array([1.5, 1.5, 5, 5, 3.14, 5, 0, 0])

    env = gym.make('LunarLander-v2')
    total_timesteps = 100000

    model = PPO_SNN(env=env,
                    save_model=False,
                    record_ghost=False,
                    record_output=False,
                    save_dir=False,
                    record_wandb=False,
                    train_config=None,
                    **SNN_hyperparameters,
                    **hyperparameters)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    actor, actor_state_dict = model.get_actor()

    # Evaluate the model
    mean_reward = eval_policy(actor,
                              env,
                              n_eval_episodes=15,
                              num_steps=SNN_hyperparameters["num_steps"],
                              threshold=threshold,
                              shift=shift)


    return mean_reward  # Metric for optuna


# Function to initialize the study and create the database
def initialize_study():
    study = optuna.create_study(direction="maximize", storage="sqlite:///./optuna_snn_study.db",
                                study_name="snn_optimization", load_if_exists=True)
    print("Database and study initialized!")


# Run the study in parallel processes
def run_study():
    study = optuna.create_study(direction="maximize", storage="sqlite:///./optuna_snn_study.db",
                                study_name="snn_optimization", load_if_exists=True)
    study.optimize(objective, n_trials=3)


if __name__ == "__main__":
    # Initialize the study (creates the database and tables)
    initialize_study()

    # Use multiprocessing to run parallel trials
    import multiprocessing

    #n_parallel_workers = multiprocessing.cpu_count()  # Adjust based on available resources
    n_parallel_workers = 4

    processes = []

    for _ in range(n_parallel_workers):
        p = multiprocessing.Process(target=run_study)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Load the study to retrieve the best result
    study = optuna.load_study(study_name="snn_optimization", storage="sqlite:///./optuna_snn_study.db")
    print("Best hyperparameters:", study.best_params)
