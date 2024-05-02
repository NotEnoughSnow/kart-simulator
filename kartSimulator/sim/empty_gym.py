import math
import os
from typing import Union, Optional

import numpy as np
import pygame_gui
import pymunk
import pymunk.pygame_util
import torch
from gym import spaces
import gym
from pymunk import Vec2d

import kartSimulator.sim.utils as utils
import kartSimulator.sim.LIDAR_vision as vision


import pygame

from kartSimulator.sim.utils import normalize_vec

PPM = 100

BOT_SIZE = 0.192
BOT_WEIGHT = 1

# 1 meter in 4.546 sec or 0.22 meters in 1 sec
# at 0.22 m/s, the bot should walk 1 meter in 4.546 seconds
# MAX_VELOCITY = 0.22 * PPM

MAX_VELOCITY = 2 * PPM

# 0.1 rad/frames, 2pi in 1.281 sec

# example : 0.0379 rad/frames, for frames = 48
# or        1.82 rad/sec
# RAD_VELOCITY = 2.84

RAD_VELOCITY = 10


ANGLE_DIFF = -math.pi * 1/2

window_width = 1500
window_length = 1000


ui_start_x = 1000

accelerate_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'accelerate.png'))
not_accelerate_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'not_accelerate.png'))
break_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'break.png'))
not_break_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'not_break.png'))


class KartSim(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60, "name": "kart2D"}

    def __init__(self, render_mode=None, train=False, obs_type="position"):

        self.render_mode = render_mode
        self.obs_type = obs_type

        speed = 1.0

        if not train:
            speed = 1.0

        print(speed)

        if render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window_surface = pygame.display.set_mode((window_width, window_length))

            # UI
            self._guiManager = pygame_gui.UIManager((window_width, window_length))
            self._guiManager.set_visual_debug_mode(True)

            gui_window = pygame_gui.elements.UIWindow(rect=pygame.rect.Rect((ui_start_x, 0), (500, 1000)),
                                                      window_display_title='',
                                                      manager=self._guiManager)

            self.gui_text_velocity = pygame_gui.elements.UILabel(relative_rect=pygame.rect.Rect((50, 300), (500, 100)),
                                                                 text="",
                                                                 container=gui_window,
                                                                 manager=self._guiManager)

            self.break_ui = pygame_gui.elements.UIImage(relative_rect=pygame.Rect((50, 0), (100, 100)),
                                                        container=gui_window,
                                                        image_surface=not_break_image,
                                                        manager=self._guiManager)

            self.accelerate_ui = pygame_gui.elements.UIImage(relative_rect=pygame.Rect((250, 0), (100, 100)),
                                                             container=gui_window,
                                                             image_surface=not_accelerate_image,
                                                             manager=self._guiManager)

            self._draw_options = pymunk.pygame_util.DrawOptions(self._window_surface)

            # clock
            self._clock = pygame.time.Clock()

        self.FPS = 100

        self._space = pymunk.Space()
        # self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = speed / 60.0
        self._t = 0

        self._background = pygame.Surface((window_width, window_length))
        self._background.fill(pygame.Color("black"))

        self._playerShape = None
        self._playerBody = None
        self._steerAngle = 0
        self.position = (0, 0)
        self.velocity = 0

        # FIXME restore shapes to (-1,1), (0,1), (0,1)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32
        )
        # do nothing, left, right, gas, brake

        # TODO use variables
        self.observation_space = spaces.Box(
            low=-400, high=400, shape=(1 + 2 * 2,), dtype=np.float32
            # 2 points for vision ray count + position, angles, velocity
        )

        self.initial_pos = 300, 450
        self.initial_angle = 0 + ANGLE_DIFF
        self.vision_points = []
        self.vision_lengths = []
        self.break_value = 0
        self.accel_value = 0
        self.steer_right_value = 0
        self.steer_left_value = 0

        # reward variables
        self.reward = 0.0
        self.prev_reward = 0.0

        # track values
        self.out_of_track = False
        self.finish = False
        self.sector_info = {}
        self._num_sectors = 1
        self._last_sector_time = 0
        # change for lap
        self.next_sector_name = "sector 1"
        self.next_target_distance = 0

        self._create_ball()
        self._add_static_scenery()

    def step(self, action: Union[np.ndarray, int]):

        if self.render_mode == "human":
            self.render(self.render_mode)

            # if i assign this to FPS, both of them become 0
            # what?
            # self.FPS = self._clock.get_fps()
            # print("what ", self._clock.get_fps())

        break_value = 0
        accel_value = 0
        steer_right_value = 0
        steer_left_value = 0

        if action is not None:
            # [0] does nothing

            steer_right_value = self._steer_right(action[1])
            steer_left_value = self._steer_left(action[2])
            accel_value = self._accelerate(action[3])
            break_value = self._break(action[4])

        self.break_value = break_value
        self.accel_value = accel_value
        self.steer_right_value = steer_right_value
        self.steer_left_value = steer_left_value

        # TODO step based on FPS
        self._space.step(self._dt)
        # TODO lap and time counters

        # FIXME move angle calculations to a diff place
        # FIXME angle changes when car moving only and based on car's vel
        angle_diff = self._steerAngle * 0.2
        self._playerBody.angle += angle_diff

        x = self._playerBody.velocity[0] * math.cos(angle_diff) - self._playerBody.velocity[1] * math.sin(
            angle_diff)
        y = self._playerBody.velocity[0] * math.sin(angle_diff) + self._playerBody.velocity[1] * math.cos(
            angle_diff)

        self._playerBody.velocity = (x, y)

        self._steerAngle /= 3
        self._playerBody.velocity /= 1.005

        # TODO stopping speed
        # TODO max speed

        # observation
        if self.obs_type == "LIDAR":
            state = self.observation_LIDAR()
        if self.obs_type == "position":
            state = self.observation_position()

        step_reward = 0
        terminated = False
        truncated = False

        # FIXME should look better
        next_target_distance = (self._playerBody.position - self.sector_info[self.next_sector_name][1]).__abs__()
        self.next_target_distance = np.exp(4 - (next_target_distance / 200)) - 20

        if action is not None:  # First step without action, called from reset()
            self.reward -= 1
            self.reward += self.next_target_distance

            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # TODO assign sector and lap based rewards

            # TODO if finish lap then truncated
            if self.finish:
                terminated = True
                step_reward += 1000

            # TODO if collide with track then terminate
            if self.out_of_track:
                truncated = True
                step_reward = -50

        self._t += 1

        if (self._playerBody.angle - ANGLE_DIFF) > (2 * math.pi):
            self._playerBody.angle = 0 + ANGLE_DIFF
        if (self._playerBody.angle - ANGLE_DIFF) < -(2 * math.pi):
            self._playerBody.angle = 0 + ANGLE_DIFF

        # print( self._playerBody.position)

        # (shape[0][0] + shape[1][0]) / 2


        return state, step_reward, terminated, truncated, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset()
        self.out_of_track = False
        self.finish = False
        self.sector_info = {}
        self.break_value = 0
        self.accel_value = 0
        self.steer_value = 0
        self.reward = 0
        self.prev_reward = 0
        self._playerBody.position = self.initial_pos
        self._playerBody.velocity = 0, 0
        self._playerBody.angle = self.initial_angle
        self._t = 0
        self._last_sector_time = 0

        self._init_sectors(self._sector_midpoints)
        self.next_sector_name = "sector 1"

        if self.obs_type == "LIDAR":
            observation = self.observation_LIDAR()
        if self.obs_type == "position":
            observation = self.observation_position()

        # return self.step(None)[0], {}
        return observation, {}

    def render(self, mode):

        pygame.display.update()
        # pygame.display.flip()

        # TODO figure out time
        time_delta = self._clock.tick(60)
        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

        # resetting screen
        self._window_surface.blit(self._background, (0, 0))
        # self._window_surface.fill(pygame.Color("black"))

        # updating events
        self._process_events()
        self._guiManager.update(time_delta)

        # drawing
        self._draw_objects()
        self._guiManager.draw_ui(self._background)

        vision.draw_cone(self._window_surface, self._playerBody)
        vision.draw_rays(self._window_surface, Vec2d(ui_start_x + 250, 850), self.vision_points, 0.3, True, True)

        self._draw_UI_icons(self.accel_value,
                            self.break_value,
                            self.steer_right_value,
                            self.steer_left_value)

        # return self.isopen

        self.gui_text_velocity.set_text(f"rewards from target : {self.next_target_distance:.4f}")

    def close(self):
        if self._window_surface is not None:
            pygame.display.quit()
            # fixme isOpen variable
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


    def _add_static_scenery(self) -> None:
        static_body = self._space.static_body

        shapes_arr = utils.readTrackFile("kartSimulator\\sim\\resources\\boxes.txt")
        sectors_arr = utils.readTrackFile("kartSimulator\\sim\\resources\sectors_box.txt")

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

        self._space.add(*static_lines)

        # sectors
        sensor_bodies = self._space.static_body

        static_sector_lines = []
        sector_midpoints = []

        for shape in sectors_arr:
            static_sector_lines.append(pymunk.Segment(sensor_bodies, shape[0], shape[1], 0.0))
            # FIXME use np.average ?
            sector_midpoints.append([(shape[0][0] + shape[1][0]) / 2, (shape[0][1] + shape[1][1]) / 2])

        self._sector_midpoints = sector_midpoints

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].elasticity = 0
            static_sector_lines[i].friction = 1
            static_sector_lines[i].sensor = True

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].collision_type = i + 2
            static_sector_lines[i].filter = pymunk.ShapeFilter(categories=0x10)

        self._space.add(*static_sector_lines)

        # track collision
        track_col = self._space.add_collision_handler(0, 1)
        track_col.begin = self.track_callback_begin
        track_col.separate = self.track_callback_end
        track_col.pre_solve = self.track_callback_isTouch

        num_sectors = 0
        # sectors collision
        for i in range(len(static_sector_lines)):
            col = self._space.add_collision_handler(0, i + 2)
            col.data["number"] = i + 1
            col.begin = self.sector_callback
            num_sectors += 1

        self._num_sectors = num_sectors

        self._init_sectors(sector_midpoints)

    def _init_sectors(self, sector_midpoints):

        for i in range(1, self._num_sectors + 1):
            self.sector_info["sector " + str(i)] = []
            self.sector_info["sector " + str(i)].append(0)
            self.sector_info["sector " + str(i)].append(sector_midpoints[i - 1])

    def _draw_UI_icons(self, acc_value, break_value, steer_right_value, steer_left_value):

        if not break_value:
            self.break_ui.set_image(not_break_image)
        else:
            self.break_ui.set_image(break_image)

        if not acc_value:
            self.accelerate_ui.set_image(not_accelerate_image)
        else:
            self.accelerate_ui.set_image(accelerate_image)

    def _steer_right(self, value):
        """steering control

        :param value: (0..1)
        :return:
        """
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            self._steerAngle += (RAD_VELOCITY / self.FPS) * value
            return value

    def _steer_left(self, value):
        """steering control

        :param value: (-1..1)
        :return:
        """
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            self._steerAngle -= (RAD_VELOCITY / self.FPS) * value
            return value

    def _accelerate(self, value):
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
                self._playerBody.apply_impulse_at_local_point((0, 2 * value), (0, 0))
            return value

    def _break(self, value):
        """breaking control

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
                self._playerBody.apply_impulse_at_local_point((0, -2 * value), (0, 0))
            return value

    def _create_ball(self) -> None:
        mass = BOT_WEIGHT
        radius = BOT_SIZE * PPM
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = self.initial_pos
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0
        shape.friction = 1
        body.angle = self.initial_angle
        shape.collision_type = 0

        self._space.add(body, shape)
        self._playerShape = shape
        self._playerBody = body

    def _draw_objects(self) -> None:
        self._space.debug_draw(self._draw_options)

    def sector_callback(self, arbiter, space, data):

        name = "sector " + str(data["number"])

        # sets the next milestone name, limited by number of sectors
        if self._num_sectors >= data["number"] + 1:
            self.next_sector_name = "sector " + str(data["number"] + 1)

        if self.sector_info.get(name)[0] == 0:
            # print("visited " + name + " for the first time")
            time_diff = self._t - self._last_sector_time
            self.sector_info[name][0] = time_diff
            self._last_sector_time = self._t
            self.reward += self._calculate_reward(time_diff)

            if data["number"] == self._num_sectors:
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

    def observation_position(self):
        obs = []

        # position
        self.position = self._playerBody.position

        # velocity
        # get velocity
        self.velocity = self._playerBody.velocity.__abs__()
        velocity = self.velocity
        # normalize and clip velocity
        velocity = np.clip(
            np.abs(normalize_vec([velocity],
                                 maximum=MAX_VELOCITY,
                                 minimum=0)),
            a_max=1,
            a_min=0)

        # rotation
        # normalize steer_angle
        steer_angle = normalize_vec([self._steerAngle], 0.0142, -0.0142)

        # assign rotations
        rotation = [self._playerBody.angle - ANGLE_DIFF, steer_angle[0]]


        return np.concatenate(
            [self.position]
            + [[self.velocity]]
            + [rotation]
        )

    def observation_LIDAR(self):
        obs = []

        # velocity
        # get velocity
        self.velocity = self._playerBody.velocity.__abs__()
        velocity = self.velocity
        # normalize and clip velocity
        velocity = np.clip(
            np.abs(normalize_vec([velocity],
                                 maximum=MAX_VELOCITY,
                                 minimum=0)),
            a_max=1,
            a_min=0)

        # rotation
        # normalize steer_angle
        steer_angle = normalize_vec([self._steerAngle], 0.0142, -0.0142)

        # assign rotations
        rotation = [self._playerBody.angle - ANGLE_DIFF, steer_angle[0]]

        # LIDAR vision
        # collect vision rays
        self.vision_points, vision_lengths = vision.cast_rays_lengths(self._space,
                                                                      self._playerBody)
        # apply circularity and convolution
        vision_lengths = vision.process_vision(vision_lengths)

        # normalize rays
        vision_lengths = normalize_vec(vision_lengths, maximum=vision.maximum, minimum=vision.minimum)

        self.vision_lengths = vision_lengths

        return np.concatenate(
            [velocity]
            + [rotation]
            + [self.vision_lengths]
        )
