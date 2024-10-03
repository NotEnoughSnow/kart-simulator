import math
import os
from typing import Union, Optional

import numpy as np
import pymunk
import pymunk.pygame_util
import torch
from gym import spaces
import gym

import kartSimulator.sim.utils as utils
import kartSimulator.sim.LIDAR_vision as vision

import pygame

from kartSimulator.sim.utils import normalize_vec

from kartSimulator.sim.maps.map_generator import MapGenerator
from kartSimulator.sim.maps.map_loader import MapLoader
from kartSimulator.sim.maps.random_point import RandomPoint

from kartSimulator.sim.ui_manager import UImanager

from kartSimulator.core.agent import Agent, Simple_agent

from kartSimulator.sim.maps.track_factory import TrackFactory

PPM = 100

BOT_SIZE = 0.192
BOT_WEIGHT = 1

# 1 meter in 4.546 sec or 0.22 meters in 1 sec
# at 0.22 m/s, the bot should walk 1 meter in 4.546 seconds
MAX_VELOCITY = 4 * 0.22 * PPM

# MAX_VELOCITY = 2 * PPM

# 0.1 rad/frames, 2pi in 1.281 sec

# example : 0.0379 rad/frames, for frames = 48
# or        1.82 rad/sec
RAD_VELOCITY = 2.84

# RAD_VELOCITY = 10

MAX_TARGET_DISTANCE = 600

window_width = 1500
window_length = 1000

WORLD_CENTER = [500, 500]



class KartSim(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60, "name": "kart2D"}

    def __init__(self,
                 num_agents=[1],
                 colors=[(255,0,0,255)],
                 obs_seq=[],
                 reset_time=300,
                 track_type="default",
                 track_args=None,
                 player_args=None,
                 ):

        print("loaded env:", self.metadata["name"])

        self.reset_time = reset_time
        self.obs_seq = obs_seq
        self.obs_len = 0

        #for item in obs_seq:
        #    self.obs_len += item[1]

        speed = 1.0

        print("game speed:", speed)

        pygame.init()
        pygame.display.init()


        self.screen = pygame.display.set_mode((window_width, window_length))
        self.surface = pygame.Surface((window_width, window_length))

        self.ui_manager = UImanager(self.screen, window_width, window_length)

        self._draw_options = pymunk.pygame_util.DrawOptions(self.screen)


        self._clock = pygame.time.Clock()

        self.FPS = 100

        self._space = pymunk.Space()
        # self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = speed / 60.0
        self._current_episode_time = 0

        self.surface = pygame.Surface((window_width, window_length))
        self.surface.fill(pygame.Color("black"))

        self.forward_direction = 0
        # FIXME obs returned before calculating the actual distance
        self.distance_to_next_points = -MAX_TARGET_DISTANCE
        self.distance_to_next_points_vec = [-MAX_TARGET_DISTANCE, -MAX_TARGET_DISTANCE]

        # FIXME restore shapes to (-1,1), (-1,1)
        #self.action_space = spaces.Box(
        #    low=-1, high=1, shape=(2,), dtype=np.float32
        #)
        # up_down, left_right

        # TODO use variables
        #self.observation_space = spaces.Box(
        #    low=-1000, high=1000, shape=(self.obs_len,), dtype=np.float32
        #)

        self.initial_pos = 300, 450
        self.vision_points = []
        self.vision_lengths = []

        # track values
        self._num_sectors = 1

        # change for lap
        self.next_target_rew = 0
        self.next_target_rew_act = 0

        self.agent_array = []

        print(num_agents)
        print(colors)
        #assert len(num_agents) == len(colors), "num_agents and colors must be of the same length"

        self.agent_count = sum(num_agents)
        self.num_dead_agents = 0
        self.num_alive_agents = 0

        for agent_count, color in zip(num_agents, colors):
            for _ in range(agent_count):
                self.agent_array.append(Simple_agent(color))


        self.map = TrackFactory.create_track(track_type,
                                             self._space,
                                             WORLD_CENTER,
                                             **track_args)

        #self.map = MapLoader(self._space, "boxes.txt", "sectors_box.txt", self.initial_pos)
        #self.map = MapGenerator(self._space, WORLD_CENTER, 50)
        #self.map = RandomPoint(self._space, spawn_range=400, wc=WORLD_CENTER)

        # map walls
        # sector initiation
        # TODO part of the world class
        self._init_world()
        self._init_sectors(self._sector_midpoints)

        position_array = []
        for i in range(len(self.get_agents())):
            position_array.append(self.initial_pos)

        print(np.shape(position_array))

        self.norm_dist = 1
        self.norm_dist_vec = [1, 1]

        self.goal_pos = [0, 0]
        self.angle_to_target = 0

        self.current_batch = 1
        self.max_num_batches = 1

        self.info = {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset()

        directions, _, position = self.map.reset([])

        self.num_alive_agents = 0
        self.num_dead_agents = 0

        agents = self.get_agents()

        position_array = []
        for i in range(len(agents)):
            position_array.append(position)
            agents[i].dead = False

        self._current_episode_time = 0

        self._init_world()
        self._init_sectors(self._sector_midpoints)


        # TODO reset should reseet the agent count accordingly


    def step(self, position_array: Union[np.ndarray, int], current_batch, max_num_batches):

        #print(len(position_array))
        #print(self.agent_count)

        self.current_batch = current_batch
        self.max_num_batches = max_num_batches

        assert len(position_array) == self.agent_count

        agents = self.get_agents()

        # need to iterate on all actions for all agents
        for i in range(len(agents)):
            if not agents[i].dead:
                agents[i].position = (position_array[i][0], position_array[i][1])

                if agents[i].position == (0,0):
                    agents[i].dead = True
                    agents[i].position = (-100, -100)
                    self.num_dead_agents += 1


        self._space.step(self._dt)

        self._current_episode_time += 1

        self.render(self.render_mode)

        self.info["fps"] = self._clock.get_fps()

        if self.next_sector_name is not None:
            self.goal_pos = self.sector_info[self.next_sector_name][1]

        self.num_alive_agents = self.agent_count - self.num_dead_agents

        done = False

        if self.num_alive_agents == 0:
            done = True

        return done




    def get_agents(self):
        return self.agent_array

    def render(self, mode):


        # TODO time pocket
        #pygame.display.update()

        time_delta = self._clock.tick(60)
        pygame.display.set_caption("timestep: " + str(self._current_episode_time))

        # resetting screen
        self.screen.blit(self.surface, (0, 0))
        # self._window_surface.fill(pygame.Color("black"))
        self.surface.fill(pygame.Color("black"))

        # drawing debug objects
        self._space.debug_draw(self._draw_options)

        # update ui
        # TODO figure out which ui stays
        self.update_ui(time_delta)

        # updating events
        self._process_events()

        self.draw_agents()
        self.draw_goal()

        pygame.display.flip()


    def draw_agents(self):
        agents = self.get_agents()

        for agent in agents:
            pygame.draw.circle(self.screen, agent.color, agent.position, agent.radius, 1)

    def draw_goal(self):
        pygame.draw.circle(self.screen, (255, 255, 0, 255), self.goal_pos, 20, 1)

    def update_ui(self, time_delta):
        self.ui_manager.update(time_delta, self.surface)

        #self.ui_manager.draw_vision_cone(self._playerBody)

        #self.ui_manager.draw_vision_points(self.vision_points)
        #self.ui_manager.draw_UI_icons(self.accel_value,
        #                              self.break_value,
        #                              self.steer_right_value,
        #                              self.steer_left_value)

        # TODO remove or replace
        self.ui_manager.add_ui_text("Current batch", f"{self.current_batch}/{self.max_num_batches}", "")
        self.ui_manager.add_ui_text("Num agents", f"{self.num_alive_agents}", "")
        self.ui_manager.add_ui_text("Current episode time", self._current_episode_time, "")


    def close(self):
        pygame.display.quit()
        pygame.quit()

    def _process_events(self) -> None:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self.screen, "screenshots/kart_replays.png")
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
            num_sectors += 1

        self._num_sectors = num_sectors

    def _init_world(self):
        self._add_walls()
        self._add_sectors()
        self.out_of_track = False


    def _init_sectors(self, sector_midpoints):
        self.finish = False
        self.sector_info = {}
        self._last_sector_time = 0
        self.next_sector_name = "sector 1"

        for i in range(1, self._num_sectors + 1):
            self.sector_info["sector " + str(i)] = []
            self.sector_info["sector " + str(i)].append(0)
            self.sector_info["sector " + str(i)].append(sector_midpoints[i - 1])

