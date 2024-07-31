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

from kartSimulator.core.agent import Agent

window_width = 1500
window_length = 1000

WORLD_CENTER = [500, 500]


class CustomDrawOptions(pymunk.pygame_util.DrawOptions):
    def __init__(self, surface):
        super().__init__(surface)

    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        alpha = 128  # Set the alpha value (0-255), where 255 is fully opaque and 0 is fully transparent
        fill_color = (255, 0, 0, 120)  # Modify the fill color to include alpha value

        p = pymunk.pygame_util.to_pygame(pos, self.surface)
        radius = int(radius)

        pygame.draw.circle(self.surface, fill_color, pos, radius)
        # pygame.gfxdraw.filled_circle(self.surface, int(p.x), int(p.y), radius, fill_color)
        # pygame.gfxdraw.aacircle(self.surface, int(p.x), int(p.y), radius, outline_color)


class KartSim(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60, "name": "kart2D"}

    def __init__(self, num_agents=1, render_mode=None, train=False, obs_seq=[]):

        print("loaded env:", self.metadata["name"])

        self.render_mode = "human"

        self.obs_seq = obs_seq
        self.obs_len = 0

        for item in obs_seq:
            self.obs_len += item[1]

        print(self.obs_len)

        speed = 1.0

        if not train:
            speed = 1.0

        print("game speed:", speed)

        pygame.init()
        pygame.display.init()

        self._window_surface = pygame.display.set_mode((window_width, window_length))
        self._background = pygame.Surface((window_width, window_length), pygame.SRCALPHA)

        self._draw_options = CustomDrawOptions(self._background)
        # self._draw_options = pymunk.pygame_util.DrawOptions(self._window_surface)

        self._clock = pygame.time.Clock()

        self.FPS = 100

        self._space = pymunk.Space()
        # self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = speed / 60.0
        self._current_episode_time = 0

        self.forward_direction = 0

        self.agent_array = []
        self.agent_count = 1

        for i in range(num_agents):
            self.agent_array.append(Agent(self._space, "replay"))

        # self.map = MapLoader(self._space, "boxes.txt", "sectors_box.txt", self.initial_pos)
        # self.map = MapGenerator(self._space, WORLD_CENTER, 50)
        self.map = RandomPoint(self._space, spawn_range=400, wc=WORLD_CENTER)


        position_array = []
        for i in range(len(self.get_agents())):
            position_array.append([500,500])

        print(np.shape(position_array))
        self._init_players(position_array)



    def step(self, action_array: Union[np.ndarray, int]):

        agents = self.get_agents()

        self._space.step(self._dt)

        self._current_episode_time += 1

        self.render("human")


    def get_agents(self):
        return self.agent_array

    def render(self, mode):
        self._clock.tick(60)
        self._window_surface.fill(pygame.Color("black"))
        self._window_surface.blit(self._background, (0, 0))

        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

        # resetting screen
        # self._window_surface.fill(pygame.Color("black"))

        # drawing debug objects
        self._space.debug_draw(self._draw_options)

        # update ui
        # TODO figure out which ui stays
        # self.update_ui(time_delta)

        # updating events
        self._process_events()
        pygame.display.flip()


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



    def _init_players(self, position_array):

        agents = self.get_agents()

        for i in range(len(agents)):
            agents[i].init_vars(position_array[i])


kart = KartSim()

while True:
    kart.step(None)