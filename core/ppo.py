import gym
import torch
from torch.distributions import MultivariateNormal
from torch.optim.adam import Adam


class PPO:



    def __init__(self, policy_class, env, **hyperparameters):

        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)
        assert (type(env.action_space) == gym.spaces.Box)

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)  # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # TODO values
        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_timesteps):


        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        #fixme should be based on timesteps not iterations
        while i_so_far < total_timesteps:
            self.rollout()
            i_so_far += 1

    def rollout(self):

        t = 0
        obs = self.env.reset()[0]

        while t < self.timesteps_per_batch:
            #print(t)
            t += 1  # Increment timesteps ran this batch so far

            # Calculate action and make a step in the env.
            # Note that rew is short for reward.
            action, log_prob = self.get_action(obs)
            print(action)

            obs, rew, terminated, truncated, _ = self.env.step(None)

            # If the environment tells us the episode is terminated, break
            if terminated or truncated:
                break

    def evaluate(self):
        pass

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 200
        #self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None  # Sets the seed of our program, used for reproducibility of results
