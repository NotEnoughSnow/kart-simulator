

from kartSimulator.core.replay_ghosts import ReplayGhosts



class Replay:


    def __init__(self, track_type, track_name, env_factory):

        # Parameters for replays
        replay_files = [f"./saves/projects/testing-project/ANN-1/ghost.hdf5"]
        mode = "all"

        ReplayGhosts(locations=replay_files,
                     mode=mode,
                     track_type=track_type,
                     track_name=track_name,
                     env_factory=env_factory,
                     )
