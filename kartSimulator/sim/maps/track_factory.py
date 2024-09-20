from kartSimulator.sim.maps.map_generator import MapGenerator
from kartSimulator.sim.maps.map_loader import MapLoader
from kartSimulator.sim.maps.random_point import RandomPoint

class TrackFactory:
    @staticmethod
    def create_track(track_type, space, world_center, **track_args):
        """
        Creates the appropriate track based on the track type and additional parameters.

        Parameters:
        - track_type: (str) The type of the track (e.g., 'boxes', 'generator', 'random').
        - space: (object) The physics space object.
        - world_center: (tuple) The center of the world in the environment.
        - initial_pos: (tuple) The initial position for the map loader.
        - track_params: Additional parameters specific to the track type.
        """

        if track_type == "boxes":
            return MapLoader(space, track_args.get('boxes_file', 'boxes.txt'),
                             track_args.get('sectors_file', 'sectors_box.txt'),
                             track_args.get('initial_pos', [300, 450])
                             )
        elif track_type == "generator":
            return MapGenerator(space,
                                world_center,
                                track_args.get('corridor_size', 50),
                                )
        elif track_type == "simple_goal":
            return RandomPoint(space,
                               spawn_range=track_args.get('spawn_range', 400),
                               fixed_goal=track_args.get('fixed_goal', [200,200]),
                               wc=world_center)
        else:
            raise ValueError(f"Unknown track type: {track_type}")