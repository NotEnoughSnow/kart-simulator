import random

import pymunk

from kartSimulator.sim.maps.map_manager import AbstractMap as abs_map


class EmptyMap(abs_map):

    def __init__(self, space, spawn_range, wc):
        self.space = space
        self.spawn_range = spawn_range
        self.wc = wc.copy()

    def reset(self, playerShape):

        for item in self.space.shapes:
            if item != playerShape:
                self.space.remove(item)

        position = self.wc

        angle = 0

        return None, angle, position

    def create_walls(self):
        return []

    def create_goals(self, mode="random"):
        if mode == "random":
            goal_x = random.uniform(-self.spawn_range / 2, self.spawn_range / 2)
            goal_y = random.uniform(-self.spawn_range / 2, self.spawn_range / 2)

            random_goal = [self.wc[0] + goal_x, self.wc[1] + goal_y]

        if mode == "static":
            random_goal = [self.wc[0] - 200, self.wc[1] - 200]


        # random_goal = [600, 600]

        # FIXME make maps
        # sectors
        sensor_bodies = self.space.static_body

        static_sector_lines = []
        sector_midpoints = []

        # static_sector_lines.append(pymunk.Segment(sensor_bodies, goal[0], goal[1], 0.0))
        shape = pymunk.Segment(sensor_bodies, random_goal, random_goal, 5.0)
        shape.color = (255, 255, 0, 255)
        static_sector_lines.append(shape)
        sector_midpoints.append(random_goal)

        # FIXME use np.average ?
        # FIXME midpoints

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].elasticity = 0
            static_sector_lines[i].friction = 1
            static_sector_lines[i].sensor = True

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].collision_type = i + 2
            static_sector_lines[i].filter = pymunk.ShapeFilter(categories=0x10)

        return static_sector_lines, sector_midpoints
