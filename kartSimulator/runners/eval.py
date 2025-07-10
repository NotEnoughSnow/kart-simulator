from kartSimulator.core.modules import env_factory
from kartSimulator.core.modules.eval_module import Evaluator


class Eval:


    def __init__(self, track_type, track_name, env_factory, save_config):


        save_config["save_dir"] = save_config["save_dir"] + "projects/"

        env = env_factory.createEnv(track_type, track_name, "human")
        #env = gym.make('LunarLander-v2', render_mode="human")

        actor_state = f"./saves/projects/spiky-turtle/base-6/ppo_actor.pth"


        NType = "SNN"

        evaluator = Evaluator(env, actor_state, NType=NType)

        if NType == "ANN":
            mean_rew = evaluator.eval_policy_ANN(n_eval_episodes=5)
        else:
            mean_rew = evaluator.eval_policy_SNN(n_eval_episodes=5)


        #evaluator.eval_sb3(env)





