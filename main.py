import kartSim.kartSim as kart_sim
from core.network import FeedForwardNN
from core.ppo import PPO

def train(env, hyperparameters):



    model = PPO(env=env, policy_class=FeedForwardNN, **hyperparameters)

    #load actor / critic

    model.learn(total_timesteps=10000)

def test():

    #init policy

    #load model

    #eval policy
    pass


def main():

    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
    hyperparameters = {
        'timesteps_per_batch': 2048,
        'max_timesteps_per_episode': 200,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 3e-4,
        'clip': 0.2,
        'render': True,
        'render_every_i': 10
    }

    env = kart_sim.KartSim(render_mode="human")
    train(env=env, hyperparameters=hyperparameters)

if __name__ == "__main__":
    main()