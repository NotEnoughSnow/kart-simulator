import os

from kartSimulator.sim import calibrate_new
from kartSimulator.sim import calibrate_new_2

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from kartSimulator.runners.train import Train
from kartSimulator.runners.train_curr import Train_curr
from kartSimulator.runners.eval import Eval
from kartSimulator.runners.replay import Replay
from kartSimulator.runners.play import Play

from kartSimulator.core.arguments import get_args

import kartSimulator.sim.steer_env as steer_env
import kartSimulator.sim.steer_gazebo as steer_gazebo
import kartSimulator.sim.grid_env as grid_env
from kartSimulator.core.modules.env_factory import EnvFactory

def main(args):

    # TODO craft saving struct
    # TODO project name, run name
    # TODO config wandb to save in it
    # Save parameters
    # experiment_name : change to test out different conditions

    description = (f"one agent died near the finish line because of the reset time. setting it to 2k")

    save_config = {
        "project_name": "spiky-turtle",
        "run_name": "base",
        "save_dir": "./saves/",
        "description": description,
    }

    # grid_env
    # steer_env
    # calibrate_new
    # calibrate_new_2
    # steer_gazebo
    env_name = steer_gazebo
    track_type = "loader"
    track_name = "big_S"


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
        Replay(track_type=track_type, track_name=track_name, env_factory=env_factory, save_config=save_config)


if __name__ == "__main__":
    args = get_args()

    # you can also directly set the args
    # args.mode = "train"
    # modes : play, train, eval, replay

    args.mode = "replay"

    main(args)
