import glob
import os
import time

import numpy as np
from stable_baselines3 import PPO
#import supersuit as ss
from stable_baselines3.ppo import MlpPolicy

import kartSimulator.sim.empty as empty_sim


def train(env_fn, steps: int = 100, seed: int = 0, **env_kwargs):
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.KartSim(render_mode="human", manual=False, train=True)

    env.reset()

    #print(f"Starting training on {str(env.metadata['name'])}.")
    #env = ss.multiagent_wrappers.pad_observations_v0(env)
    #env = ss.pettingzoo_env_to_vec_env_v1(env)
    #env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Model
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        batch_size=256,
        learning_rate=0.001,
        ent_coef=0.001,
    )
    #     model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, batch_size=256,)
    #     model = DQN("MlpPolicy", env, verbose=1, batch_size=256,)

    # Train
    model.learn(total_timesteps=steps, progress_bar=True)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def eval(env_fn, num_games: int = 100, render_mode: str = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)

    #     model = RecurrentPPO.load(latest_policy)
    #     model = DQN.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if agent == 'agent_0':
                obs = np.append(obs, [0, 0])
            # print(obs)
            if render_mode == 'human':
                time.sleep(0.01)
            for agent in env.agents:
                rewards[agent] += env.rewards[agent]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }

    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward

env_kwargs = dict(num_agents=2, num_hostages=1, num_goals=0, max_cycles=25, continuous_actions=False)
env_fn = empty_sim
train(env_fn, steps=50000, seed=0, render_mode=None, **env_kwargs)
#eval(env_fn, num_games=10, render_mode='human', **env_kwargs)
