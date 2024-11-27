import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from kartSimulator.runners.train import Train
from kartSimulator.runners.train_curr import Train_curr
from kartSimulator.runners.eval import Eval
from kartSimulator.runners.replay import Replay
from kartSimulator.runners.play import Play

from kartSimulator.core.arguments import get_args


import kartSimulator.sim.env.grid_env as grid_env
from kartSimulator.core.modules.env_factory import EnvFactory

def main(args):

    # TODO craft saving struct
    # TODO project name, run name
    # TODO config wandb to save in it
    # Save parameters
    # experiment_name : change to test out different conditions


    save_config = {
        "project_name": "grid-final",
        "run_name": "SNN",
        "save_dir": "./saves/",
    }

    env_name = grid_env
    track_type = "loader"
    track_name = "small_S"

    env_factory = EnvFactory(env_name)


    if args.mode == "train":
        Train(track_type=track_type, track_name=track_name, env_factory=env_factory, save_config=save_config)
    if args.mode == "train_curr":
        Train_curr(track_type=track_type, env_factory=env_factory, save_config=save_config)
    if args.mode == "play":
        Play(track_type=track_type, track_name=track_name, env_factory=env_factory, save_config=save_config)
    if args.mode == "eval":
        Eval(track_type=track_type, track_name=track_name, env_factory=env_factory, save_config=save_config)
    if args.mode == "replay":
        Replay(track_type=track_type, track_name=track_name, env_factory=env_factory)



if __name__ == "__main__":
    args = get_args()

    # you can also directly set the args
    # args.mode = "train"
    # modes : play, train, eval, replay

    args.mode = "train"

    main(args)
