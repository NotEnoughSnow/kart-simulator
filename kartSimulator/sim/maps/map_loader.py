from kartSimulator.sim.maps.map_manager import AbstractMap as abs_map
import pymunk
import kartSimulator.sim.utils as utils


class MapLoader(abs_map):

    def __init__(self, space,  track_file, sector_file, initial_pos):
        self.missing_walls_flag = True
        self.missing_sectors_flag = True

        self.space = space
        self.sector_name = sector_file
        self.track_name = track_file
        self.initial_pos = initial_pos

    def reset(self, playerShape):
        angle = 0
        position = self.initial_pos

        return None, angle, position

    def create_walls(self):
        static_body = self.space.static_body

        shapes_arr = utils.readTrackFile("kartSimulator\\sim\\resources\\"+self.track_name)

        static_lines = []

        for shape in shapes_arr:
            for i in range(len(shape) - 1):
                static_lines.append(pymunk.Segment(static_body, shape[i], shape[i + 1], 0.0))

        for line in static_lines:
            line.elasticity = 0
            line.friction = 1
            line.sensor = True
            line.collision_type = 1
            line.filter = pymunk.ShapeFilter(categories=0x1)

        self.missing_walls_flag = False

        return static_lines

    def create_goals(self, mode):
        sectors_arr = utils.readTrackFile("kartSimulator\\sim\\resources\\" + self.sector_name)

        # sectors
        sensor_bodies = self.space.static_body

        static_sector_lines = []
        sector_midpoints = []

        for shape in sectors_arr:
            static_sector_lines.append(pymunk.Segment(sensor_bodies, shape[0], shape[1], 0.0))
            # FIXME use np.average ?
            sector_midpoints.append([(shape[0][0] + shape[1][0]) / 2, (shape[0][1] + shape[1][1]) / 2])

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].elasticity = 0
            static_sector_lines[i].friction = 1
            static_sector_lines[i].sensor = True

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].collision_type = i + 2
            static_sector_lines[i].filter = pymunk.ShapeFilter(categories=0x10)

        self.missing_sectors_flag = False

        return static_sector_lines, sector_midpoints
