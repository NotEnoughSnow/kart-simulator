import gymnasium as gym
from kartSimulator.core.ppo import PPO
import numpy as np
import torch

from torch.distributions import Categorical
from kartSimulator.sim.simple_env import KartSim
import kartSimulator.sim.observation_types as obs_types

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


if __name__ == "__main__":

    hyperparameters = {
        'timesteps_per_batch': 1024,
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

    total_timesteps = 1000

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

    print(num_finishes)
    print(highest)
    print(mean_reward)

