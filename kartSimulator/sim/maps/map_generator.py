import math
import random
import pymunk
from kartSimulator.sim.maps.map_manager import AbstractMap as abs_map


class Cube:

    def __init__(self, position, wc, corr_size):
        self.cc = wc.copy()
        self.walls = {}
        self.corridor_size = corr_size
        self.position = position

        if position == "CENTER":
            pass
        if position.find("LEFT") != -1:
            self.cc[0] = wc[0] - self.corridor_size * 2
        if position.find("RIGHT") != -1:
            self.cc[0] = wc[0] + self.corridor_size * 2
        if position.find("UP") != -1:
            self.cc[1] = wc[0] - self.corridor_size * 2
        if position.find("DOWN") != -1:
            self.cc[1] = wc[0] + self.corridor_size * 2

        # print(self.cc)

        self.walls["UP"] = [[self.cc[0] - self.corridor_size, self.cc[1] - self.corridor_size],
                            [self.cc[0] + self.corridor_size, self.cc[1] - self.corridor_size]]
        self.walls["RIGHT"] = [[self.cc[0] + self.corridor_size, self.cc[1] - self.corridor_size],
                               [self.cc[0] + self.corridor_size, self.cc[1] + self.corridor_size]]
        self.walls["DOWN"] = [[self.cc[0] + self.corridor_size, self.cc[1] + self.corridor_size],
                              [self.cc[0] - self.corridor_size, self.cc[1] + self.corridor_size]]
        self.walls["LEFT"] = [[self.cc[0] - self.corridor_size, self.cc[1] + self.corridor_size],
                              [self.cc[0] - self.corridor_size, self.cc[1] - self.corridor_size]]

        # print(self.walls["UP"])

    def activate(self, directions):
        active_walls = []

        right_cond = ((self.position == "CENTER" and directions[0] != "RIGHT") or
                      (self.position == "DOWN" and directions[0] == "DOWN" and directions[1] != "RIGHT") or
                      (self.position == "UP" and directions[0] == "UP" and directions[1] != "RIGHT") or
                      (self.position == "UPLEFT" and directions[0] == "LEFT" and directions[1] == "UP") or
                      (self.position == "DOWNLEFT" and directions[0] == "LEFT" and directions[1] == "DOWN") or
                      (self.position == "DOWNRIGHT" and directions[0] == "RIGHT" and directions[1] == "DOWN") or
                      (self.position == "UPRIGHT" and directions[0] == "RIGHT" and directions[1] == "UP") or
                      (self.position == "RIGHT" and directions[0] == "RIGHT" and directions[1] != "RIGHT"))

        left_cond = ((self.position == "CENTER" and directions[0] != "LEFT") or
                     (self.position == "DOWN" and directions[0] == "DOWN" and directions[1] != "LEFT") or
                     (self.position == "UP" and directions[0] == "UP" and directions[1] != "LEFT") or
                     (self.position == "UPLEFT" and directions[0] == "LEFT" and directions[1] == "UP") or
                     (self.position == "DOWNLEFT" and directions[0] == "LEFT" and directions[1] == "DOWN") or
                     (self.position == "DOWNRIGHT" and directions[0] == "RIGHT" and directions[1] == "DOWN") or
                     (self.position == "UPRIGHT" and directions[0] == "RIGHT" and directions[1] == "UP") or
                     (self.position == "LEFT" and directions[0] == "LEFT" and directions[1] != "LEFT"))

        up_cond = ((self.position == "CENTER" and directions[0] != "UP") or
                     (self.position == "UP" and directions[0] == "UP" and directions[1] != "UP") or
                     (self.position == "UPLEFT" and directions[0] == "UP" and directions[1] == "LEFT") or
                     (self.position == "LEFT" and directions[0] == "LEFT" and directions[1] != "UP") or
                     (self.position == "UPRIGHT" and directions[0] == "UP" and directions[1] == "RIGHT") or
                     (self.position == "RIGHT" and directions[0] == "RIGHT" and directions[1] != "UP") or
                     (self.position == "DOWNRIGHT" and directions[0] == "DOWN" and directions[1] == "RIGHT") or
                     (self.position == "DOWNLEFT" and directions[0] == "DOWN" and directions[1] == "LEFT"))

        down_cond = ((self.position == "CENTER" and directions[0] != "DOWN") or
                     (self.position == "DOWN" and directions[0] == "DOWN" and directions[1] != "DOWN") or
                     (self.position == "DOWNLEFT" and directions[0] == "DOWN" and directions[1] == "LEFT") or
                     (self.position == "LEFT" and directions[0] == "LEFT" and directions[1] != "DOWN") or
                     (self.position == "DOWNRIGHT" and directions[0] == "DOWN" and directions[1] == "RIGHT") or
                     (self.position == "RIGHT" and directions[0] == "RIGHT" and directions[1] != "DOWN") or
                     (self.position == "UPRIGHT" and directions[0] == "UP" and directions[1] == "RIGHT") or
                     (self.position == "UPLEFT" and directions[0] == "UP" and directions[1] == "LEFT"))

        if right_cond:
            active_walls.append(self.walls["RIGHT"])

        if left_cond:
            active_walls.append(self.walls["LEFT"])

        if up_cond:
            active_walls.append(self.walls["UP"])

        if down_cond:
            active_walls.append(self.walls["DOWN"])

        return active_walls

class MapGenerator(abs_map):
    cubes = []

    def __init__(self, space, world_center, corr_size):

        self.corridor_size = corr_size
        self.space = space

        self.wc = world_center.copy()

        self.create_cubes()

        self.directions = ["UP", "DOWN"]

    def create_cubes(self):
        self.cubes.append(Cube("CENTER", self.wc, self.corridor_size))

        self.cubes.append(Cube("LEFT", self.wc, self.corridor_size))
        self.cubes.append(Cube("RIGHT", self.wc, self.corridor_size))
        self.cubes.append(Cube("UP", self.wc, self.corridor_size))
        self.cubes.append(Cube("DOWN", self.wc, self.corridor_size))

        self.cubes.append(Cube("UPLEFT", self.wc, self.corridor_size))
        self.cubes.append(Cube("UPRIGHT", self.wc, self.corridor_size))
        self.cubes.append(Cube("DOWNRIGHT", self.wc, self.corridor_size))
        self.cubes.append(Cube("DOWNLEFT", self.wc, self.corridor_size))

    def get_walls(self, directions):
        walls_to_build = []

        for cube in self.cubes:
            for wall in cube.activate(directions):
                walls_to_build.append(wall)

        return walls_to_build

    def get_goals(self, directions):
        goals = []

        if directions == ["UP", "RIGHT"]:
            goals.append([[self.wc[0] + self.corridor_size * 3, self.wc[1] - self.corridor_size * 3],
                          [self.wc[0] + self.corridor_size * 3, self.wc[1] - self.corridor_size]])

        if directions == ["RIGHT", "RIGHT"]:
            goals.append([[self.wc[0] + self.corridor_size*3, self.wc[1] - self.corridor_size],
                            [self.wc[0] + self.corridor_size*3, self.wc[1] + self.corridor_size]])

        if directions == ["DOWN", "RIGHT"]:
            goals.append([[self.wc[0] + self.corridor_size * 3, self.wc[1] + self.corridor_size],
                          [self.wc[0] + self.corridor_size * 3, self.wc[1] + self.corridor_size * 3]])

        if directions == ["RIGHT", "DOWN"]:
            goals.append([[self.wc[0] + self.corridor_size * 3, self.wc[1] + self.corridor_size * 3],
                          [self.wc[0] + self.corridor_size, self.wc[1] + self.corridor_size * 3]])

        if directions == ["DOWN", "DOWN"]:
            goals.append([[self.wc[0] + self.corridor_size, self.wc[1] + self.corridor_size*3],
                            [self.wc[0] - self.corridor_size, self.wc[1] + self.corridor_size*3]])

        if directions == ["LEFT", "DOWN"]:
            goals.append([[self.wc[0] - self.corridor_size, self.wc[1] + self.corridor_size * 3],
                          [self.wc[0] - self.corridor_size * 3, self.wc[1] + self.corridor_size * 3]])

        if directions == ["DOWN", "LEFT"]:
            goals.append([[self.wc[0] - self.corridor_size * 3, self.wc[1] + self.corridor_size * 3],
                          [self.wc[0] - self.corridor_size * 3, self.wc[1] + self.corridor_size]])

        if directions == ["LEFT", "LEFT"]:
            goals.append([[self.wc[0] - self.corridor_size*3, self.wc[1] + self.corridor_size],
                            [self.wc[0] - self.corridor_size*3, self.wc[1] - self.corridor_size]])

        if directions == ["UP", "LEFT"]:
            goals.append([[self.wc[0] - self.corridor_size*3, self.wc[1] - self.corridor_size],
                            [self.wc[0] - self.corridor_size*3, self.wc[1] - self.corridor_size*3]])

        if directions == ["LEFT", "UP"]:
            goals.append([[self.wc[0] - self.corridor_size * 3, self.wc[1] - self.corridor_size*3],
                          [self.wc[0] - self.corridor_size, self.wc[1] - self.corridor_size * 3]])

        if directions == ["UP", "UP"]:
            goals.append([[self.wc[0] - self.corridor_size, self.wc[1] - self.corridor_size*3],
                            [self.wc[0] + self.corridor_size, self.wc[1] - self.corridor_size*3]])

        if directions == ["RIGHT", "UP"]:
            goals.append([[self.wc[0] + self.corridor_size, self.wc[1] - self.corridor_size * 3],
                          [self.wc[0] + self.corridor_size * 3, self.wc[1] - self.corridor_size * 3]])

        return goals

    def generate_random_direction(self):
        # Define all possible directions
        all_directions = ["UP", "DOWN", "LEFT", "RIGHT"]

        all_pairs = [[d1, d2] for d1 in all_directions for d2 in all_directions]


        # Exclude the forbidden pairs
        excluded_pairs = [["DOWN", "UP"], ["UP", "DOWN"], ["LEFT", "RIGHT"], ["RIGHT", "LEFT"]]
        valid_pairs = [pair for pair in all_pairs if pair not in excluded_pairs]


        # Randomly choose a valid direction
        directions = random.choice(valid_pairs)

        # Set player's starting angle based on the first direction
        if directions[0] == "UP":
            player_angle = 3 * math.pi / 2
        elif directions[0] == "DOWN":
            player_angle = math.pi / 2
        elif directions[0] == "LEFT":
            player_angle = math.pi
        else:  # "RIGHT"
            player_angle = 0


        player_pos = self.wc.copy()

        pos_variation_x = random.uniform(-self.corridor_size/2, self.corridor_size/2)
        pos_variation_y = random.uniform(-self.corridor_size/2, self.corridor_size/2)

        random_position = [player_pos[0]+pos_variation_x, player_pos[1]+pos_variation_y]

        # Return player's starting angle
        return directions, player_angle, random_position

    def reset(self, playerShape):

        for item in self.space.shapes:
            if item != playerShape:
                self.space.remove(item)

        directions, angle, position = self.generate_random_direction()

        self.directions = directions

        return directions, angle, position

    def create_walls(self):
        static_body = self.space.static_body

        static_lines = []

        walls_to_build = self.get_walls(self.directions)

        for wall in walls_to_build:
            #static_lines.append(pymunk.Segment(static_body, wall[0], wall[1], 0.0))
            shape = pymunk.Segment(static_body, wall[0], wall[1], 0.0)
            #shape.color = (255, 255, 255, 255)
            static_lines.append(shape)

        return static_lines

    def create_goals(self):
        # FIXME make maps
        # sectors
        sensor_bodies = self.space.static_body

        static_sector_lines = []
        sector_midpoints = []

        goals_to_set = self.get_goals(self.directions)

        if len(goals_to_set) == 0:
            static_sector_lines.append(pymunk.Segment(sensor_bodies, [0,0], [0,0], 0.0))
            sector_midpoints.append([0, 0])

        for goal in goals_to_set:
            #static_sector_lines.append(pymunk.Segment(sensor_bodies, goal[0], goal[1], 0.0))
            shape = pymunk.Segment(sensor_bodies, goal[0], goal[1], 0.0)
            shape.color = (255, 255, 0, 255)
            static_sector_lines.append(shape)
            sector_midpoints.append([(goal[0][0] + goal[1][0]) / 2, (goal[0][1] + goal[1][1]) / 2])


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