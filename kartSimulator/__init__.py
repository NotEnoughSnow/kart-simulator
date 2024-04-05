from gym.envs.registration import register

register(
    id='kart2D-v0',
    entry_point='kartSimulator.sim:empty_gym',
    max_episode_steps=300,
)