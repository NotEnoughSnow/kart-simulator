import math
import os
from typing import Union, Optional
import time

import numpy as np
import pymunk
import pymunk.pygame_util
import torch
from gymnasium import spaces
import gymnasium as gym

import kartSimulator.sim.utils as utils
from kartSimulator.sim.LIDAR_vision import LIDAR_vision

import pygame

from kartSimulator.sim.utils import normalize_vec, normalize_vec_unsymmetric

from kartSimulator.sim.maps.track_factory import TrackFactory

from kartSimulator.sim.ui_manager import UImanager

# TODO changes to kart speed
# max_velocity, burgerbot = 0.22
# impulse, old = 2
# rad vel = 2.84

PPM = 100

MAX_TARGET_DISTANCE = 250

ANGLE_DIFF = -math.pi * 1 / 2

window_width = 1500
window_length = 1000


WORLD_CENTER = [500, 500]


class Robot:
    def __init__(self, player_max_velocity, player_rad_velocity):
        self.player_max_velocity = player_max_velocity
        self.player_rad_velocity = player_rad_velocity
        self.input_velocity = 0
        self.input_angle = 0

        self.pending_command = (0, 0)
        self.linear_timer = 0
        self.angular_right_timer = 0
        self.angular_left_timer = 0


    def input_command(self, action_array):

        if action_array[0] == 1:
            self.pending_command = (0, 0)
        elif action_array[1] == 1:
            # self.input_velocity = 1 * self.player_max_velocity * PPM
            # self.input_angle = 0
            self.pending_command = (1, 0)
            self.linear_timer += 2
        elif action_array[2] == 1:
            # self.input_velocity = 0
            # self.input_angle = -1
            self.pending_command = (0, -1)
            self.angular_left_timer += 2
        elif action_array[3] == 1:
            # self.input_velocity = 0
            # elf.input_angle = 1
            self.pending_command = (0, 1)
            self.angular_right_timer += 2

        self.linear_timer -= 1
        self.angular_right_timer -= 1
        self.angular_left_timer -= 1


    def update(self, body):


        self.linear_timer = np.clip(a=self.linear_timer, a_min=0, a_max=25)
        self.angular_right_timer = np.clip(a=self.angular_right_timer, a_min=0, a_max=25)
        self.angular_left_timer = np.clip(a=self.angular_left_timer, a_min=0, a_max=25)

        # This runs every frame
        #if time.time() - self.command_applied_time >= self.command_delay:
        # Only now we apply the input
        linear_input, angular_input = self.pending_command

        #if self.linear_timer > 10:
        self.input_velocity = linear_input * self.player_max_velocity * PPM

        #if self.angular_right_timer > 10 and angular_input>0:
        self.input_angle = angular_input

        if self.angular_left_timer > 10 and angular_input<0:
            self.input_angle = angular_input

        if linear_input == 0:
            self.input_velocity = 0

        if angular_input == 0:
            self.input_angle = 0

        self.pending_command = None  # Clear pending

        # FIXME move angle calculations to a diff place
        # FIXME angle changes when car moving only and based on car's vel

        x = self.input_velocity * math.cos(body.angle - ANGLE_DIFF)
        y = self.input_velocity * math.sin(body.angle - ANGLE_DIFF)

        body.angular_velocity = self.input_angle * self.player_rad_velocity

        body.velocity = (x, y)

        # TODO stopping speed
        # TODO max speed


class KartSim(gym.Env):
    metadata = {"render_modes": [None, "human"],
                "render_fps": 60,
                "name": "kart2D steer_gazebo",
                "track": "***",
                "obs_seq": [],
                "reset_time": 2000,
                }

    def __init__(self,
                 render_mode=None,
                 train=False,
                 obs_seq=[],
                 reset_time=2000,
                 track=None,
                 player_args=None,
                 rew_adj=None,
                 ):

        print("loaded env: ", self.metadata["name"])

        self.render_mode = render_mode
        self.metadata["reset_time"] = reset_time
        self.metadata["obs_seq"] = obs_seq

        # player stuff
        self.player_max_velocity = player_args["player_acc_rate"]
        self.player_rad_velocity = player_args["rad_velocity"]
        self.bot_size = player_args["bot_size"]
        self.bot_weight = player_args["bot_weight"]

        self.reset_time = reset_time
        self.obs_seq = obs_seq
        self.obs_len = 0


        print(self.obs_len)

        speed = 1.0

        if not train:
            speed = 1.0

        print("game speed:", speed)

        self.vision = LIDAR_vision(vision_length=player_args["vision_length"], vision_fov=player_args["vision_fov"], ray_count=player_args["vision_ray_count"])

        if render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window_surface = pygame.display.set_mode((window_width, window_length))

            self.ui_manager = UImanager(self._window_surface, window_width, window_length, self.vision)

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
        self.distance_to_next_goal = -MAX_TARGET_DISTANCE
        self.distance_to_next_goal_vec = [-MAX_TARGET_DISTANCE, -MAX_TARGET_DISTANCE]

        self.norm_dist = 1
        self.norm_dist_vec = [1, 1]

        low = []
        high = []

        for obs_type in obs_seq:
            if len(obs_type) == 2:
                self.obs_len += obs_type[1]

            if len(obs_type) == 3:
                self.obs_len += len(obs_type[1])

                for item_low in obs_type[1]:
                    low.append(item_low)

                for item_high in obs_type[2]:
                    high.append(item_high)
            if len(obs_type) == 4:
                self.obs_len += obs_type[1]

                for i in range(obs_type[1]):
                    low.append(-obs_type[2][0])
                    high.append(obs_type[3][0])


        #self.low = np.array(low).astype(np.float32)
        #self.high = np.array(high).astype(np.float32)

        #self.observation_space = spaces.Box(self.low, self.high)
        
        self.observation_space = spaces.Box(low=-1, high=1, shape= (self.obs_len,), dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        # do nothing, accelerate, break, steer_left, steer_right


        self.initial_angle = 0 + ANGLE_DIFF
        self.vision_points = []
        self.vision_lengths = []

        # track values
        self._num_sectors = 1

        # change for lap
        self.next_target_rew = 0
        self.next_target_rew_act = 0

        #self.map = MapLoader(self._space, "boxes.txt", "sectors_1.txt", self.initial_pos)
        #self.map = MapGenerator(self._space, WORLD_CENTER, 50)
        #self.map = RandomPoint(self._space, spawn_range=400, wc=WORLD_CENTER)

        self.map = track
        track.init_track(self._space, WORLD_CENTER)


        self.initial_pos = self.map.initial_pos

        self.create_player()

        self.robot = Robot(self.player_max_velocity, self.player_rad_velocity)

        # map walls
        # sector initiation
        # TODO part of the world class
        self._init_world()
        self._init_sectors(self._sector_midpoints)
        self._init_player(self.initial_pos, 0)


        self.next_goal_position = [0, 0]

        self.angles = {}
        self.angles["current_angle"] = 0
        self.angles["angle_to_target"] = 0
        self.angles["angle_to_target_cos"] = 0
        self.angles["angle_to_target_sin"] = 0

        self.info = {}

        self.highest_goal = 0
        self.num_finishes = 0

        self.standing_still_timesteps = 0

        self.continuous = False

        # epsilon variables
        self.epsilon = 0.7
        self.epsilon_lowest_dist = math.inf

        self.resets = 0

        self.steer_reward = 0

        self.rew_adj = rew_adj

        self.input_velocity = 0
        self.input_angle = 0

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):

        self.resets += 1

        super().reset()

        import_position = options.get("initial pos", None)

        directions, angle, position = self.map.reset([self._playerShape])

        self._current_episode_time = 0

        self._init_world()
        self._init_sectors(self._sector_midpoints)

        if import_position is None:
            pass
        else:
            position = import_position

        angle = math.pi/2

        self._init_player(position, angle)

        observation = self.observation()

        #print(self.resets)

        info = {"player pos": position}

        # return self.step(None)[0], {}
        return observation, info

    def step(self, action: Union[np.ndarray, int]):

        if not self.continuous:
            action_array = np.zeros(self.action_space.n)
            action_array[action] = 1

        #print(action)
        #print(action_array)


        pstart = self._playerBody.position

            # if i assign this to FPS, both of them become 0
            # what?
            # self.FPS = self._clock.get_fps()
            # print("what ", self._clock.get_fps())


        steer_value = 0
        accel_break_value = 0

        #print(action)

        if action_array is not None:

            self.robot.input_command(action_array)

            #accel_value = self._accelerate(action_array[1])
            #break_value = self._break(action_array[2])
            #steer_left_value = self._steer_left(action_array[3])
            #steer_right_value = self._steer_right(action_array[4])

        self.accel_value = 0
        self.break_value = 0

        self.steer_left_value = 0
        self.steer_right_value = 0

        # TODO step based on FPS
        self._space.step(self._dt)
        # TODO lap and time counters

        pend = self._playerBody.position

        self.robot.update(self._playerBody)

        direction = (math.cos(self._playerBody.angle - ANGLE_DIFF), math.sin(self._playerBody.angle - ANGLE_DIFF))

        self.forward_direction = direction[0] * self._playerBody.velocity[0] + direction[1] * self._playerBody.velocity[
            1]

        self.velocity = self._playerBody.velocity.__abs__()

        step_reward = 0
        terminated = False
        truncated = False

        self.check_standing_still(25, 160)
        self.check_deserting(30)


        if action is not None:

            self.reward_logic(pstart, pend, action)

            step_reward, terminated, truncated = self.reward_function()

        self._current_episode_time += 1

        # fix angle observation, limit it in 0 - 2pi range
        if (self._playerBody.angle - ANGLE_DIFF) > (2 * math.pi):
            self._playerBody.angle = 0 + ANGLE_DIFF
        if (self._playerBody.angle - ANGLE_DIFF) < -(2 * math.pi):
            self._playerBody.angle = 0 + ANGLE_DIFF

        # print( self._playerBody.position)

        # (shape[0][0] + shape[1][0]) / 2

        self.calculate_angle_to_goal()

        # observation
        state = self.observation()

        # truncation
        if self._current_episode_time > self.reset_time:
            self.out_of_track = True
            pass

        if self.render_mode == "human":
            self.render(self.render_mode)

        self.info["fps"] = self._clock.get_fps()
        self.info["position"] = self._playerBody.position
        self.info["highest"] = self.highest_goal
        self.info["num_finishes"] = self.num_finishes


        #if self.highest_goal == 1:

        #    if self.distance_to_next_points < self.epsilon_lowest_dist:
        #        self.epsilon_lowest_dist = self.distance_to_next_points

        #    self.epsilon = self.epsilon_lowest_dist/180

        #self.info["epsilon"] = self.epsilon

        self.epsilon = 0.7 - self.resets / 1400

        self.info["epsilon"] = np.clip(self.epsilon, a_min=0, a_max=1)

        return state, step_reward, terminated, truncated, self.info

    def calculate_angle_to_goal(self):
        x = self.next_goal_position[0] - self._playerBody.position[0]
        y = self.next_goal_position[1] - self._playerBody.position[1]

        magnitude = math.sqrt(x ** 2 + y ** 2)

        self.angles["angle_to_target_cos"] = x / magnitude
        self.angles["angle_to_target_sin"] = y / magnitude


    def check_standing_still(self, threshold=15, max_still_timesteps=200):
        if np.abs(self.velocity) < threshold:  # Velocity close to zero
            self.standing_still_timesteps += 1
        else:
            self.standing_still_timesteps = 0  # Reset if moving

        # If agent has been still for too long, truncate the episode
        if self.standing_still_timesteps >= max_still_timesteps:
            #print("cut")
            self.out_of_track = True
            self.standing_still_timesteps = 0

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

    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def calculate_speed_factor(self, x):
        return math.exp(x / 100) / 2

    def calculate_distance_rew(self, x, s):
        return 2000 / (x + 50) - 10

    def calculate_steer_reward(self, direction):
        return direction*0.1

    def draw_angles(self, screen):
        # Get player position
        player_position = self._playerBody.position

        # Length of the lines (in pixels)
        line_length = 100

        # Calculate the endpoint of the line for the angle to the target (blue line)
        target_x = player_position[0] + line_length * math.cos(self.angles["angle_to_target"])
        target_y = player_position[1] + line_length * math.sin(self.angles["angle_to_target"])

        # Draw the blue line representing the angle to the target
        pygame.draw.line(screen, (0, 0, 255), player_position, (target_x, target_y), 2)

        # Calculate the endpoint of the line for the player's current angle (green line)
        player_x = player_position[0] + line_length * math.cos(self.angles["current_angle"])
        player_y = player_position[1] + line_length * math.sin(self.angles["current_angle"])

        # Draw the green line representing the player's current angle
        pygame.draw.line(screen, (0, 255, 0), player_position, (player_x, player_y), 2)
    def steer_direction(self, action):
        # Angle to target using arctan2 to get the angle from sine and cosine
        self.angles["angle_to_target"] = math.atan2(self.angles["angle_to_target_sin"], self.angles["angle_to_target_cos"])

        # Get the current orientation of the ball
        self.angles["current_angle"] = self._playerBody.angle - 3*math.pi/2

        # Calculate the angle difference
        angle_diff = self.angles["angle_to_target"] - self.angles["current_angle"]

        # Normalize the angle to be within -pi to +pi
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Check if the goal is to the right or left
        if angle_diff > 0:
            # Goal is to the left of the player
            if action == 2:
                direction = -2
            elif action == 3:
                direction = 1
            else:
                direction = 0
        else:
            # Goal is to the right of the player
            if action == 3:
                direction = -2
            elif action == 2:
                direction = 1
            else:
                direction = 0

        return direction

    def reward_logic(self, pstart, pend, action):

        if self.next_sector_name is not None:

            self.next_goal_position = self.sector_info[self.next_sector_name][1]

            distance_to_next_points_vec = pend - self.next_goal_position
            self.distance_to_next_goal_vec = [-abs(distance_to_next_points_vec[0]),
                                              -abs(distance_to_next_points_vec[1])]

            self.distance_to_next_goal = self.distance(pend, self.next_goal_position)

            # distance based reward. the reward is a direct transformation applied to the distance
            # from the next goal to the agent.
            # with added bonus, the transformation can be sector dependant.
            #target_number = int(self.next_sector_name[-1])
            self.next_target_rew = self.calculate_distance_rew(self.distance_to_next_goal, 0)

            # calculates a speed factor to be used as part of the dist-act reward
            speed_factor = self.calculate_speed_factor(self.velocity)

            # dist-act reward
            # rewards/penalizes the agent if it moves closer/farther in the current timestep, i.e. if the difference in
            # distance is less/more since the start of the current timestep.
            initial_distance = -self.distance(self.next_goal_position, pstart)
            final_distance = -self.distance(self.next_goal_position, pend)
            self.next_target_rew_act = (final_distance - initial_distance)*speed_factor

            # steer reward
            if action is not None:
                self.steer_reward = self.calculate_steer_reward(self.steer_direction(action))


    def reward_function(self):

        step_reward = 0
        terminated = False
        truncated = False

        # penelty for existing
        self.reward -= 1 * self.rew_adj["passive"]

        # distance reward
        self.reward += self.next_target_rew * self.rew_adj["dist"]

        # action-distance based reward
        self.reward += (self.next_target_rew_act / 80) * self.rew_adj["act_dist"]

        # reward for crossing sectors. reset the value after.
        self.reward += self.sector_time_reward * self.rew_adj["sector_time"]
        self.sector_time_reward = 0

        # TODO aligning reward
        self.reward += self.steer_reward * self.rew_adj["steer"]
        self.steer_reward = 0

        # if finish lap then truncated
        if self.finish:
            terminated = True
            self.reward += 1000

        # if collide with track then terminate
        if self.out_of_track:
            truncated = True
            self.reward -= 500

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

        # draw angles of agent, and direction of goal
        self.draw_angles(self._window_surface)



    def update_ui(self, time_delta):
        self.ui_manager.update(time_delta, self._background)

        self.ui_manager.draw_vision_cone(self._playerBody)

        self.ui_manager.draw_vision_points(self.vision_points)
        self.ui_manager.draw_UI_icons(self.accel_break_value,
                                      self.steer_value,)

        self.ui_manager.add_ui_text("next target", self.next_sector_name, "")
        self.ui_manager.add_ui_text("distance to target", self.distance_to_next_goal, ".4f")
        self.ui_manager.add_ui_text("norm dist", self.norm_dist, ".3f")
        self.ui_manager.add_ui_text("total reward", self.reward, ".3f")
        self.ui_manager.add_ui_text("act.rew from target", self.next_target_rew_act, ".3f")
        self.ui_manager.add_ui_text("current angle", self.angles["current_angle"], ".4f")
        self.ui_manager.add_ui_text("angle to target", self.angles["angle_to_target"], ".4f")
        self.ui_manager.add_ui_text("steer rew.", self.steer_reward, ".4f")

        self.ui_manager.add_ui_text("time in sec", (pygame.time.get_ticks() / 1000), ".2f")
        self.ui_manager.add_ui_text("fps", self._clock.get_fps(), ".2f")
        self.ui_manager.add_ui_text("steps", self._current_episode_time, ".2f")

        self.ui_manager.add_ui_text("velocity", self.velocity, ".2f")
        self.ui_manager.add_ui_text("position x", self._playerBody.position[0], ".0f")
        self.ui_manager.add_ui_text("position y", self._playerBody.position[1], ".0f")

        self.ui_manager.add_ui_text("norm dist vec x", self.norm_dist_vec[0], ".3f")
        self.ui_manager.add_ui_text("norm dist vec y", self.norm_dist_vec[1], ".3f")

        self.ui_manager.add_ui_text("standing_still_timesteps", self.standing_still_timesteps, ".0f")
        self.ui_manager.add_ui_text("deserting timesteps", self.deserting_timesteps, ".0f")

        self.ui_manager.add_ui_text("epsilon", self.epsilon, ".4f")
        self.ui_manager.add_ui_text("epsilon distance", self.epsilon_lowest_dist, ".4f")


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

    def _init_player(self, position, angle):
        self.accel_break_value = 0
        self.steer_value = 0

        self.reward = 0
        self.sector_time_reward = 0
        self.prev_reward = 0

        # self._playerBody.position = self.initial_pos
        self._playerBody.position = position

        self._playerBody.velocity = 0, 0
        self._playerBody.angle = angle + ANGLE_DIFF

    def _init_sectors(self, sector_midpoints):
        self.finish = False
        self.sector_info = {}
        self._last_sector_time = 0
        self.next_sector_name = "sector 1"

        for i in range(1, self._num_sectors + 1):
            self.sector_info["sector " + str(i)] = []
            self.sector_info["sector " + str(i)].append(0)
            self.sector_info["sector " + str(i)].append(sector_midpoints[i - 1])


    def _steer_left(self, value):
        """steering control

        :param value: (0..1)
        :return:
        """
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            self.input_angle -= 1
            return value

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
            self.input_angle += 1
            return value

    """
    def _steer_left(self, value):

        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            self._steerAngle -= (RAD_VELOCITY / self.FPS) * value
            return value
    """
    """
    def _accelerate_break(self, value):
        '''acceleration control

        :param value: (-1..1)
        :return:
        '''
        # FIXME temp fix for ensuing that value is within (0,1)
        value = min(1, value)
        value = max(-1, value)

        if value == 0:
            return value
        elif value > 0:
            if self.velocity < self.max_velocity:
                self._playerBody.apply_impulse_at_local_point((0, self.player_acc_rate * value), (0, 0))
            return value
        elif value < 0:
            if self.forward_direction > 0:
                self._playerBody.apply_impulse_at_local_point((0, self.player_break_rate * value), (0, 0))
            return value
        else:
            raise Exception("how ?")
    """

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
            self.input_velocity = 1

            return value

    def _break(self, value):

        # FIXME temp fix for ensuing that value is within (0,1)
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            self.input_velocity = 0
            return value

    """"
    def _break(self, value):

        # FIXME temp fix for ensuing that value is within (0,1)
        value = min(1, value)
        value = max(0, value)

        if value == 0:
            return value
        else:
            if self.forward_direction > 0:
                self._playerBody.apply_impulse_at_local_point((0, -2 * value), (0, 0))
            return value
    """

    def create_player(self) -> None:
        mass = self.bot_weight
        radius = self.bot_size * PPM
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
            self.sector_time_reward += self._calculate_sector_time_reward(time_diff)
            #print(f"in : {time_diff} received : {self._calculate_reward(time_diff)}")

            if data["number"] == self._num_sectors:
                print("reached goal!")
                self.num_finishes += 1
                self.highest_goal = self._num_sectors
                self.finish = True

        # calculatd in step()
        #if self.highest_goal == 1:
        #    self.epsilon = 0.8

        #if self.next_sector_name == "sector 2":
        #    print("huh")

        # TODO translate this to non ooga booga numbers
        #if self.highest_goal == 2:
        #    self.epsilon = 0.1

        #if self.highest_goal == 3:
        #    self.epsilon = 0.05

        #if self.highest_goal == 4:
        #    self.epsilon = 0.005

        #if self.highest_goal == 5:
        #    self.epsilon = 0

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

    def _calculate_sector_time_reward(self, time):
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

        velocity = np.clip(normalize_vec_unsymmetric([velocity],
                                                     maximum=self.player_max_velocity,
                                                     minimum=0),
            a_max=1,
            a_min=-1)

        return velocity

    def observation_rotation(self):
        # normalize steer_angle
        player_angle = (self._playerBody.angle - ANGLE_DIFF)

        # Calculate the cosine and sine of the player's angle
        player_angle_cos = math.cos(player_angle)
        player_angle_sin = math.sin(player_angle)

        # assign rotations
        rotation = [player_angle_cos, player_angle_sin]

        # sin y = 1 when agent is pointing down

        return rotation

    def observation_target_angle(self):

        # assign rotations
        rotation = [self.angles["angle_to_target_cos"], self.angles["angle_to_target_sin"]]

        return rotation

    def observation_position(self):

        #max_pos = max(window_width, window_length)/2

        #TODO min and max positions for x and y

        x_min = 54
        y_min = 81

        x_max = 990
        y_max = 958


        position_x = normalize_vec_unsymmetric([self._playerBody.position[0]], maximum=x_max, minimum=x_min)
        position_y = normalize_vec_unsymmetric([self._playerBody.position[1]], maximum=y_max, minimum=y_min)

        return [position_x[0], position_y[0]]

    def observation_distance(self):
        distance = [utils.normalize_vec([self.distance_to_next_goal], maximum=0, minimum=-MAX_TARGET_DISTANCE)[0]]

        return distance

    def observation_distance_vec(self):

        distance = utils.normalize_vec(self.distance_to_next_goal_vec, maximum=0, minimum=-MAX_TARGET_DISTANCE)

        return distance

    def observation_LIDAR(self):
        # LIDAR vision
        # collect vision rays
        self.vision_points, vision_lengths = self.vision.cast_rays_lengths(self._space, self._playerBody)
        # apply circularity and convolution
        #wraparound_data = self.vision.apply_circularity(vision_lengths)

        # normalize rays
        vision_lengths = normalize_vec_unsymmetric(vision_lengths, maximum=self.vision.vision_upper_limit, minimum=0)

        self.vision_lengths = vision_lengths
        return self.vision_lengths

    def observation_LIDAR_CONV(self):
        # LIDAR vision
        # collect vision rays
        self.vision_points, vision_lengths = self.vision.cast_rays_lengths(self._space,
                                                                      self._playerBody)
        # apply circularity and convolution
        wraparound_data = self.vision.apply_circularity(vision_lengths)

        vision_lengths = self.vision.apply_convolution(wraparound_data)

        # normalize rays
        vision_lengths = normalize_vec(vision_lengths, maximum=self.vision.maximum, minimum=self.vision.minimum)

        self.vision_lengths = vision_lengths

        return self.vision_lengths
