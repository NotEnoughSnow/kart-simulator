import sys

import torch

from kartSimulator.core.arguments import get_args
from kartSimulator.core import eval_policy
from kartSimulator.core.network import FeedForwardNN
from kartSimulator.core.ppo import PPO
from kartSimulator.core.line_policy import L_Policy
import kartSimulator.evolutionary.core as EO

import kartSimulator.sim.sim2D as kart_sim
import kartSimulator.sim.empty as empty_sim
import kartSimulator.sim.empty_gym as empty_gym
import kartSimulator.sim_turtlebot.sim_turtlebot as turtlebot_sim
import kartSimulator.sim_turtlebot.calibrate as turtlebot_calibrate
import kartSimulator.sim_1D as oneD_sim


def play(env):
    running = False

    while not running:
        env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            s, r, terminated, truncated, _ = env.step(None)

            total_reward += r
            steps += 1

            if truncated:
                print("hit a wall")
                print(f"total rewards this ep:{total_reward}")

            if terminated:
                print(f"total rewards this ep:{total_reward}")
                # TODO times

    env.close()


def train(env, hyperparameters, actor_model, critic_model):
    model = PPO(env=env, policy_class=FeedForwardNN, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # load actor / critic

    model.learn(total_timesteps=100000)


def train_line(env, hyperparameters, actor_model, critic_model):
    model = L_Policy(env=env, policy_class=FeedForwardNN, **hyperparameters)

    model.learn(total_timesteps=10000)


def optimize():
    # TODO
    EO.run()


def test(env, actor_model):
    """
    	Tests the model.

    	Parameters:
    		env - the environment to test the policy on
    		actor_model - the actor model to load in

    	Return:
    		None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = FeedForwardNN(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.

    eval_policy.eval_policy(policy=policy, env=env, render=True)

    #eval_policy.visualize(policy,env)



def main(args):
    hyperparameters = {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 400,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    sim = None

    if args.sim == "kart":
        sim = kart_sim
    if args.sim == "empty":
        sim = empty_sim
    if args.sim == "empty_gym":
        sim = empty_gym
    if args.sim == "1D":
        sim = oneD_sim
    if args.sim == "turtlebot":
        sim = turtlebot_sim
    if args.sim == "turtlebot_calibrate":
        sim = turtlebot_calibrate

    assert sim is not None

    if args.mode == "play":
        env = sim.KartSim(render_mode="human", manual=True, train=False)
        play(env=env)
    if args.mode == "train":
        env = sim.KartSim(render_mode=None, manual=False, train=True)
        train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
    if args.mode == "optimize":
        # TODO implement evolutionary optimization
        pass
    if args.mode == "test":
        env = sim.KartSim(render_mode="human", manual=True, train=False)
        test(env, actor_model="ppo_actor.pth")


if __name__ == "__main__":
    args = get_args()

    # you can also directly set the args
    # args.mode = "train"

    args.mode = "play"
    args.sim = "empty_gym"

    main(args)
