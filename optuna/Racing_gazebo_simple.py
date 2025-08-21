from pydoc import describe

from kartSimulator.core.ppo import PPO
import numpy as np

from torch.distributions import Categorical
import kartSimulator.sim.steer_gazebo as steer_gazebo
import kartSimulator.sim.observation_types as obs_types

from kartSimulator.core.modules.env_factory import EnvFactory


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

    total_timesteps = 500000

    env_name = steer_gazebo
    env_factory = EnvFactory(env_name)

    env = env_factory.createEnv("loader", "big_S", None)

    model = PPO(env=env,
                save_model=False,
                record_ghost=False,
                record_output=False,
                save_dir=False,
                record_wandb=False,
                train_config=None,
                description = None,
                run_name = None,
                project_name = None,
                **hyperparameters)

    num_finishes, highest = model.learn(total_timesteps=total_timesteps)

    actor, actor_state_dict = model.get_actor()


    mean_reward = eval_policy(actor, env, n_eval_episodes=15)

    print("## results :")
    print("num_finishes :", num_finishes)
    print("highest :", highest)
    print("mean_reward :", mean_reward)

    result = num_finishes*2000 + highest*200 + mean_reward


