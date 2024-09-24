import math
import os
from typing import Union, Optional

import numpy as np
import pymunk
import pymunk.pygame_util
import torch
from gymnasium import spaces
import gymnasium as gym

import kartSimulator.sim.utils as utils
import kartSimulator.sim.LIDAR_vision as vision

import pygame

from kartSimulator.sim.utils import normalize_vec

from kartSimulator.sim.ui_manager import UImanager

from kartSimulator.sim.maps.track_factory import TrackFactory

# TODO changes to kart speed
# max_velocity, burgerbot = 0.22
# impulse, old = 2

PPM = 100

# 1 meter in 4.546 sec or 0.22 meters in 1 sec
# at 0.22 m/s, the bot should walk 1 meter in 4.546 seconds
#MAX_VELOCITY = 4 * 0.22 * PPM

MAX_VELOCITY = 4 * PPM

MAX_TARGET_DISTANCE = 600

window_width = 1500
window_length = 1000

WORLD_CENTER = [500, 500]


class KartSim(gym.Env):
    metadata = {"render_modes": [None, "human"],
                "render_fps": 60,
                "name": "kart2D simple_env",
                "track": "***",
                "obs_seq": [],
                "reset_time": 300,
                }

    def __init__(self, render_mode=None,
                 train=False,
                 obs_seq=[],
                 reset_time=300,
                 track_type="default",
                 track_args=None,
                 player_args=None,
                 ):

        print("loaded env:", self.metadata["name"])

        self.render_mode = render_mode

        self.metadata["reset_time"] = reset_time
        self.metadata["obs_seq"] = obs_seq

        # player stuff
        self.max_velocity = player_args["max_velocity"] * PPM
        self.player_acc_rate = player_args["player_acc_rate"]
        self.bot_size = player_args["bot_size"]
        self.bot_weight = player_args["bot_weight"]

        self.reset_time = reset_time
        self.obs_seq = obs_seq
        self.obs_len = 0

        speed = 1.0

        if not train:
            speed = 1.0

        print("game speed:", speed)

        if render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window_surface = pygame.display.set_mode((window_width, window_length))

            self.ui_manager = UImanager(self._window_surface, window_width, window_length)

            self._draw_options = pymunk.pygame_util.DrawOptions(self._window_surface)

            # clock

        self._clock = pygame.time.Clock()

        self.FPS = 100

        self._space = pymunk.Space()
        # self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = speed / 60.0
        self._current_episode_time = 0

        self._background = pygame.Surface((window_width, window_length))
        self._background.fill(pygame.Color("black"))

        self._playerShape = None
        self._playerBody = None
        self.position = (0, 0)
        self.velocity = 0
        self.forward_direction = 0
        # FIXME obs returned before calculating the actual distance
        self.distance_to_next_points = -MAX_TARGET_DISTANCE
        self.distance_to_next_points_vec = [-MAX_TARGET_DISTANCE, -MAX_TARGET_DISTANCE]

        low = []
        high = []

        for obs_type in obs_seq:
            if len(obs_type) == 3:
                self.obs_len += len(obs_type[1])

                for item_low in obs_type[1]:
                    low.append(item_low)

                for item_high in obs_type[2]:
                    high.append(item_high)
            else:
                self.obs_len += len(obs_type[2])

                for i in range(obs_type[1]):
                    low.append(-obs_type[2][0])
                    high.append(obs_type[3][0])

        self.low = np.array(low).astype(np.float32)
        self.high = np.array(high).astype(np.float32)

        self.observation_space = spaces.Box(self.low, self.high)

        self.action_space = spaces.Discrete(5)
        # do nothing, up, down, left, right

        self.vision_points = []
        self.vision_lengths = []

        # track values
        self._num_sectors = 1

        # change for lap
        self.next_target_rew = 0
        self.next_target_rew_act = 0

        #self.map = MapLoader(self._space, "boxes.txt", "sectors_box.txt", self.initial_pos)
        #self.map = MapGenerator(self._space, WORLD_CENTER, 50)
        #self.map = RandomPoint(self._space, spawn_range=400, wc=WORLD_CENTER)

        self.map = TrackFactory.create_track(track_type,
                                             self._space,
                                             WORLD_CENTER,
                                             **track_args)

        self.initial_pos = self.map.initial_pos

        self.create_player()


        # map walls
        # sector initiation
        # TODO part of the world class
        self._init_world()
        self._init_sectors(self._sector_midpoints)
        self._init_player(self.initial_pos)

        self.norm_dist = 1
        self.norm_dist_vec = [1, 1]

        self.goal_pos = [0,0]
        self.angle_to_target_sin = 0
        self.angle_to_target_cos = 0

        self.info = {}

        self.highest_goal = 0
        self.num_finishes = 0

        self.deserting_timesteps = 0

        self.continuous = False

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset()

        directions, _, position = self.map.reset([self._playerShape])

        self._current_episode_time = 0

        self._init_world()
        self._init_sectors(self._sector_midpoints)
        self._init_player(position)

        observation = self.observation()

        #print("huh ", len(self._space.shapes))

        # return self.step(None)[0], {}
        return observation, {}

    def step(self, action: Union[np.ndarray, int]):

        if not self.continuous:
            action_array = np.zeros(self.action_space.n)
            action_array[action] = 1

        self._clock.tick()

        # if i assign this to FPS, both of them become 0
        # what?
        # self.FPS = self._clock.get_fps()
        # print("what ", self._clock.get_fps())

        up_value = 0
        down_value = 0
        left_value = 0
        right_value = 0

        #print("first step :", self._playerBody.position)
        pstart = self._playerBody.position

        if action_array is not None:
            go_up_value = self._go_up(action_array[1])
            go_down_value = self._go_down(action_array[2])
            go_left_value = self._go_left(action_array[3])
            go_right_value = self._go_right(action_array[4])

        self.go_up_value = None
        self.go_down_value = None
        self.go_left_value = None
        self.go_right_value = None

        # TODO step based on FPS
        self._space.step(self._dt)
        # TODO lap and time counters

        #print("second step :", self._playerBody.position)
        pend = self._playerBody.position

        self.velocity = self._playerBody.velocity.__abs__()

        self._playerBody.velocity /= 1.005

        # TODO stopping speed
        # TODO max speed

        step_reward = 0
        terminated = False
        truncated = False

        self.check_deserting(30)

        if action is not None:
            step_reward, terminated, truncated = self.reward_function(pstart, pend)

        self._current_episode_time += 1

        # print( self._playerBody.position)

        # (shape[0][0] + shape[1][0]) / 2

        # observation
        state = self.observation()

        # truncation
        if self._current_episode_time > self.reset_time:
            self.out_of_track = True


        if self.render_mode == "human":
            self.render(self.render_mode)

        self.info["fps"] = self._clock.get_fps()
        self.info["position"] = self._playerBody.position
        self.info["highest"] = self.highest_goal


        return state, step_reward, terminated, truncated, self.info

    def check_deserting(self, max_deserting_timesteps=200):

        if self.next_target_rew_act < 0:  # going opposite of target
            self.deserting_timesteps += 1
        else:
            self.deserting_timesteps = 0  # Reset if moving

        # If agent has been still for too long, truncate the episode
        if self.deserting_timesteps >= max_deserting_timesteps:
            #print("cut")
            self.out_of_track = True
            self.deserting_timesteps = 0

    def potential_curve(self, x):
        # If x=0, then pot is maximal
        # As x decreases, pot increases monotonically
        if x >= 0:
            pot = 1.0 - x
        else:
            pot = 1.0
        return pot

    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def reward_function(self, pstart, pend):

        step_reward = 0
        terminated = False
        truncated = False


        if self.next_sector_name is not None:

            self.goal_pos = self.sector_info[self.next_sector_name][1]

            distance_to_next_points_vec = pend - self.goal_pos
            self.distance_to_next_points_vec = [-abs(distance_to_next_points_vec[0]),
                                                -abs(distance_to_next_points_vec[1])]

            self.distance_to_next_points = self.distance(pend, self.goal_pos)

            target_number = int(self.next_sector_name[-1])
            #self.next_target_rew = 2000/(self.distance_to_next_points+50) - 10

            initial_potential = self.potential_curve(self.distance(self.goal_pos, pstart))
            final_potential = self.potential_curve(self.distance(self.goal_pos, pend))
            self.next_target_rew_act = final_potential - initial_potential

            # If the agent is moving away (i.e., next_target_rew_act < 0), apply double the penalty
            if self.next_target_rew_act < 0:
                self.next_target_rew_act *= 1.5  # extra penalty for moving away

            #initial_potential = self.potential_curve(self.distance(goal, pstart))

        # penelty for existing
        # self.reward -= 1

        # reward for closing distance to sector medians
        #self.reward += self.next_target_rew
        self.reward += self.next_target_rew_act/80

        # TODO assign sector and lap based rewards

        # if finish lap then truncated
        if self.finish:
            terminated = True
            self.reward += 1000
            #step_reward = self.reward

        # if collide with track then terminate
        if self.out_of_track:
            truncated = True
            self.reward -= 500
            #step_reward = self.reward

        step_reward = self.reward

        return step_reward, terminated, truncated

    def render(self, mode):

        # TODO time pocket
        pygame.display.update()
        # pygame.display.flip()

        time_delta = self._clock.tick(60)
        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

        # resetting screen
        self._window_surface.blit(self._background, (0, 0))
        # self._window_surface.fill(pygame.Color("black"))

        # drawing debug objects
        self._space.debug_draw(self._draw_options)

        # update ui
        self.update_ui(time_delta)

        # updating events
        self._process_events()

    def update_ui(self, time_delta):
        self.ui_manager.update(time_delta, self._background)

        self.ui_manager.draw_vision_cone(self._playerBody)

        self.ui_manager.draw_vision_points(self.vision_points)
        # TODO fix this
        self.ui_manager.draw_UI_icons(0, 0)

        self.ui_manager.add_ui_text("next target", self.next_sector_name, "")
        self.ui_manager.add_ui_text("distance to target", self.distance_to_next_points, ".4f")
        self.ui_manager.add_ui_text("norm dist", self.norm_dist, ".3f")
        self.ui_manager.add_ui_text("total reward", self.reward, ".3f")
        self.ui_manager.add_ui_text("act.rew from target", self.next_target_rew_act, ".3f")
        self.ui_manager.add_ui_text("angle to target", self.angle_to_target_cos, ".3f")
        self.ui_manager.add_ui_text("angle to target", self.angle_to_target_sin, ".3f")

        self.ui_manager.add_ui_text("time in sec", (pygame.time.get_ticks() / 1000), ".2f")
        self.ui_manager.add_ui_text("fps", self._clock.get_fps(), ".2f")
        self.ui_manager.add_ui_text("steps", self._current_episode_time, ".2f")

        self.ui_manager.add_ui_text("velocity", self.velocity, ".2f")
        self.ui_manager.add_ui_text("position x", self._playerBody.position[0], ".0f")
        self.ui_manager.add_ui_text("position y", self._playerBody.position[1], ".0f")

        self.ui_manager.add_ui_text("norm dist vec x", self.norm_dist_vec[0], ".3f")
        self.ui_manager.add_ui_text("norm dist vec y", self.norm_dist_vec[1], ".3f")

        self.ui_manager.add_ui_text("deserting timesteps", self.deserting_timesteps, ".0f")


    def close(self):
        if self.render_mode is not None:
            pygame.display.quit()
        pygame.quit()

    def _process_events(self) -> None:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._window_surface, "screenshots/karts.png")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                pass

    def _add_walls(self) -> None:

        static_lines = self.map.create_walls()

        for line in static_lines:
            line.elasticity = 0
            line.friction = 1
            line.sensor = True
            line.collision_type = 1
            line.filter = pymunk.ShapeFilter(categories=0x1)

        self._space.add(*static_lines)

        # track collision
        track_col = self._space.add_collision_handler(0, 1)
        track_col.begin = self.track_callback_begin
        track_col.separate = self.track_callback_end
        track_col.pre_solve = self.track_callback_isTouch

    def _add_sectors(self) -> None:

        static_sector_lines, sector_midpoints = self.map.create_goals("static")

        self._space.add(*static_sector_lines)
        self._sector_midpoints = sector_midpoints

        # collision

        num_sectors = 0
        # sectors collision
        for i in range(len(static_sector_lines)):
            col = self._space.add_collision_handler(0, i + 2)
            col.data["number"] = i + 1
            col.begin = self.sector_callback
            num_sectors += 1

        self._num_sectors = num_sectors

    def _init_world(self):
        if self.map.missing_walls_flag:
            self._add_walls()
        if self.map.missing_sectors_flag:
            self._add_sectors()
        self.out_of_track = False

    def _init_player(self, position):
        self.break_value = 0
        self.accel_value = 0
        self.steer_right_value = 0
        self.steer_left_value = 0

        self.reward = 0
        self.prev_reward = 0

        # self._playerBody.position = self.initial_pos
        self._playerBody.position = position

        self._playerBody.velocity = 0, 0

    def _init_sectors(self, sector_midpoints):
        self.finish = False
        self.sector_info = {}
        self._last_sector_time = 0
        self.next_sector_name = "sector 1"

        for i in range(1, self._num_sectors + 1):
            self.sector_info["sector " + str(i)] = []
            self.sector_info["sector " + str(i)].append(0)
            self.sector_info["sector " + str(i)].append(sector_midpoints[i - 1])

    def _go_up(self, value):
        """acceleration control

        :param value: (0..1)
        :return:
        """
        # FIXME temp fix for ensuing that value is within (0,1)
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            if self.velocity < MAX_VELOCITY:
                self._playerBody.apply_impulse_at_local_point((0, -self.player_acc_rate * value), (0, 0))
            return value

    def _go_down(self, value):
        """acceleration control

        :param value: (0..1)
        :return:
        """
        # FIXME temp fix for ensuing that value is within (0,1)
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            if self.velocity < MAX_VELOCITY:
                self._playerBody.apply_impulse_at_local_point((0, self.player_acc_rate * value), (0, 0))
            return value

    def _go_left(self, value):
        """acceleration control

        :param value: (0..1)
        :return:
        """
        # FIXME temp fix for ensuing that value is within (0,1)
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            if self.velocity < MAX_VELOCITY:
                self._playerBody.apply_impulse_at_local_point((-self.player_acc_rate * value, 0), (0, 0))
            return value

    def _go_right(self, value):
        """acceleration control

        :param value: (0..1)
        :return:
        """
        # FIXME temp fix for ensuing that value is within (0,1)
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            if self.velocity < MAX_VELOCITY:
                self._playerBody.apply_impulse_at_local_point((self.player_acc_rate * value, 0), (0, 0))
            return value

    def create_player(self) -> None:
        mass = self.bot_weight
        radius = self.bot_size * PPM
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = self.initial_pos
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0
        shape.friction = 1
        shape.collision_type = 0

        self._space.add(body, shape)
        self._playerShape = shape
        self._playerBody = body

    def sector_callback(self, arbiter, space, data):

        name = "sector " + str(data["number"])

        # sets the next milestone name, limited by number of sectors
        if self._num_sectors >= data["number"] + 1:
            self.next_sector_name = "sector " + str(data["number"] + 1)
            if data["number"] > self.highest_goal:
                self.highest_goal = data["number"]

        if self.sector_info.get(name)[0] == 0:
            #print("visited " + name + " for the first time")
            time_diff = self._current_episode_time - self._last_sector_time
            self.sector_info[name][0] = time_diff
            self._last_sector_time = self._current_episode_time

            # reward based on sector time
            #self.reward += self._calculate_reward(time_diff)

            if data["number"] == self._num_sectors:
                print("reached goal!")
                self.num_finishes += 1
                self.highest_goal = self._num_sectors
                self.finish = True

        return True

    def track_callback_begin(self, arbiter, space, data):
        # print("exiting track")
        return True

    def track_callback_isTouch(self, arbiter, space, data):
        self.out_of_track = True
        return True

    def track_callback_end(self, arbiter, space, data):
        # print(self.touch_track_counter)
        return True

    def _calculate_reward(self, time):
        return 1 / 3 * math.exp(1 / 100 * -time + 7)

    def observation(self):
        obs_methods = {
            "LIDAR": self.observation_LIDAR,
            "LIDAR_conv": self.observation_LIDAR_CONV,
            "position": self.observation_position,
            "velocity": self.observation_velocity,
            "rotation": self.observation_rotation,
            "target_angle": self.observation_target_angle,
            "distance": self.observation_distance,
            "distance_vec": self.observation_distance_vec
        }

        obs = []
        for item in self.obs_seq:
            obs_type = item[0]
            if obs_type in obs_methods:
                res = obs_methods[obs_type]()
                obs.append(res)

        return np.concatenate([x for x in obs])

    def observation_velocity(self):
        velocity = self.velocity

        velocity = np.clip(
            np.abs(normalize_vec([velocity],
                                 maximum=self.max_velocity,
                                 minimum=0)),
            a_max=1,
            a_min=0)[0]


        return [velocity]

    def observation_rotation(self):
        raise Exception("this version doesn't use agent's rotation")

        return None

    def observation_target_angle(self):

        x = self._playerBody.position[0] - self.goal_pos[0]
        y = self._playerBody.position[1] - self.goal_pos[1]

        magnitude = math.sqrt(x ** 2 + y ** 2)

        self.angle_to_target_cos = x / magnitude
        self.angle_to_target_sin = - y / magnitude

        # assign rotations
        rotation = [self.angle_to_target_cos, self.angle_to_target_sin]

        return rotation

    def observation_position(self):

        max_pos = max(window_width, window_length)/2

        position = normalize_vec(self._playerBody.position, maximum=max_pos, minimum=0)

        return position

    def observation_distance(self):
        distance = [utils.normalize_vec([self.distance_to_next_points], maximum=0, minimum=-MAX_TARGET_DISTANCE)[0]]

        return distance

    def observation_distance_vec(self):

        distance = utils.normalize_vec(self.distance_to_next_points_vec, maximum=0, minimum=-MAX_TARGET_DISTANCE)

        return distance

    def observation_LIDAR(self):
        # LIDAR vision
        # collect vision rays
        self.vision_points, vision_lengths = vision.cast_rays_lengths(self._space,
                                                                      self._playerBody)
        # apply circularity and convolution
        wraparound_data = vision.apply_circularity(vision_lengths)

        # normalize rays
        vision_lengths = normalize_vec(wraparound_data, maximum=vision.VISION_LENGTH, minimum=0)

        self.vision_lengths = vision_lengths
        return self.vision_lengths

    def observation_LIDAR_CONV(self):
        # LIDAR vision
        # collect vision rays
        self.vision_points, vision_lengths = vision.cast_rays_lengths(self._space,
                                                                      self._playerBody)
        # apply circularity and convolution
        wraparound_data = vision.apply_circularity(vision_lengths)

        vision_lengths = vision.apply_convolution(wraparound_data)

        # normalize rays
        vision_lengths = normalize_vec(vision_lengths, maximum=vision.maximum, minimum=vision.minimum)

        self.vision_lengths = vision_lengths

        return self.vision_lengths
