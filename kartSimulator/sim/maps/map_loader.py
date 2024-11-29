import random

from kartSimulator.sim.maps.map_manager import AbstractMap as abs_map
import pymunk
import kartSimulator.sim.utils as utils


class MapLoader(abs_map):

    def __init__(self, track_file, sector_file, initial_pos, r_goal):
        self.missing_walls_flag = True
        self.missing_sectors_flag = True

        self.sector_name = sector_file
        self.track_name = track_file
        self.initial_pos = initial_pos

        self.r_goal = r_goal

    def init_track(self, space, world_center):
        self.space = space

    def reset(self, playerShapes):

        if self.track_name == "big_S_track.txt":
            pos_variation_x = random.uniform(-50, 50)
            pos_variation_y = random.uniform(-50, 50)
        else:
            pos_variation_x = random.uniform(-15, 15)
            pos_variation_y = random.uniform(-15, 15)

        random_position = [self.initial_pos[0]+pos_variation_x, self.initial_pos[1]+pos_variation_y]

        angle = 0
        position = random_position

        if self.r_goal is not None:

            for item in self.space.shapes:
                if item not in playerShapes:
                    self.space.remove(item)

            self.missing_sectors_flag = True
            self.missing_walls_flag = True


        return None, angle, position

    def create_walls(self):
        static_body = self.space.static_body

        shapes_arr = utils.readTrackFile("kartSimulator/sim/resources/"+self.track_name)

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
        sectors_arr = utils.readTrackFile("kartSimulator/sim/resources/" + self.sector_name)

        # sectors
        sensor_bodies = self.space.static_body

        static_sector_lines = []
        sector_midpoints = []

        if self.r_goal is None:
            for shape in sectors_arr:
                static_sector_lines.append(pymunk.Segment(sensor_bodies, shape[0], shape[1], 0.0))
                # FIXME use np.average ?
                sector_midpoints.append([(shape[0][0] + shape[1][0]) / 2, (shape[0][1] + shape[1][1]) / 2])

        else:

            var_x = random.uniform(self.r_goal[0][0], self.r_goal[0][1])
            var_y = random.uniform(self.r_goal[1][0], self.r_goal[1][1])

            for shape in sectors_arr:

                new_p1 = [shape[0][0] + var_x, shape[0][1] + var_y]
                new_p2 = [shape[1][0] + var_x, shape[1][1] + var_y]

                static_sector_lines.append(pymunk.Segment(sensor_bodies, new_p1, new_p2, 0.0))
                sector_midpoints.append([(new_p1[0] + new_p2[0]) / 2, (new_p1[1] + new_p2[1]) / 2])


        for i in range(len(static_sector_lines)):
            static_sector_lines[i].elasticity = 0
            static_sector_lines[i].friction = 1
            static_sector_lines[i].sensor = True

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].collision_type = i + 2
            static_sector_lines[i].filter = pymunk.ShapeFilter(categories=0x10)

        self.missing_sectors_flag = False

        return static_sector_lines, sector_midpoints
