

from kartSimulator.core.replay_ghosts import ReplayGhosts



class Replay:

    def __init__(self, track_type, track_name, env_factory, save_config):

        # extract information from save config
        project_name = save_config["project_name"]
        run_name = "base-3"

        # Parameters for replays
        replay_files = [f"./saves/projects/{project_name}/{run_name}/ghost.hdf5"]
        mode = "all" 

        ReplayGhosts(locations=replay_files,
                     mode=mode,
                     track_type=track_type,
                     track_name=track_name,
                     env_factory=env_factory,
                     )
