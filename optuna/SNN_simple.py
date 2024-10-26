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


if __name__ == "__main__":

    hyperparameters = {
        'timesteps_per_batch': 524,
        'max_timesteps_per_episode': 700,
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

    SNN_hyperparameters = {
        "num_steps": 32,
    }

    threshold = torch.tensor([1.5, 1.5, 5, 5, 3.14, 5, 1, 1])
    shift = np.array([1.5, 1.5, 5, 5, 3.14, 5, 0, 0])

    env = gym.make('LunarLander-v2')

    total_timesteps = 500

    model = PPO_SNN(env=env,
                    save_model=False,
                    record_ghost=False,
                    record_output=False,
                    save_dir=False,
                    record_wandb=False,
                    train_config=None,
                    **SNN_hyperparameters,
                    **hyperparameters)

    model.learn(total_timesteps=total_timesteps)

    actor, actor_state_dict = model.get_actor()

    print("finished")

    mean_reward = eval_policy(actor,
                              env,
                              n_eval_episodes=15,
                              num_steps=SNN_hyperparameters["num_steps"],
                              threshold=threshold,
                              shift=shift)

    print(mean_reward)

