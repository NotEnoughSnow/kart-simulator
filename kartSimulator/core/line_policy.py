


class S_Policy:


    def __init__(self):
        # Make sure the environment is compatible with our code
        assert (type(env.observation_space) == gym.spaces.Box)
        assert (type(env.action_space) == gym.spaces.Box)


