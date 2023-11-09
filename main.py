import math
import numpy as np
import random
from typing import List
import pygame
import pymunk
import pymunk.pygame_util
import csv
import ast


class Game(object):

    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        # self._space.gravity = (0.0, 900.0)

        self._space.add_collision_handler(0, 1).begin = self.start_callback
        self._space.add_collision_handler(0, 2).begin = self.end_callback

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((1000, 1000))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        self._playerShape = None
        self._playerBody = None
        self._steerAngle = 0

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

    def run(self) -> None:

        self._create_ball()

        # Main loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            # Get the state of each key
            keys = pygame.key.get_pressed()
            # key controls
            if keys[pygame.K_w]:
                self._accelerate(1)
            if keys[pygame.K_s]:
                self._break(1)
            if keys[pygame.K_d] and self._steerAngle < 1:
                self._steer(1)
            if keys[pygame.K_a] and self._steerAngle > -1:
                self._steer(-1)

            angle_diff = self._steerAngle * 0.2
            self._playerBody.angle += angle_diff

            x = self._playerBody.velocity[0] * math.cos(angle_diff) - self._playerBody.velocity[1] * math.sin(
                angle_diff)
            y = self._playerBody.velocity[0] * math.sin(angle_diff) + self._playerBody.velocity[1] * math.cos(
                angle_diff)

            self._playerBody.velocity = (x, y)

            self._steerAngle /= 3
            self._playerBody.velocity /= 1.005

            self._process_events()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def _add_static_scenery(self) -> None:
        static_body = self._space.static_body

        shapes_arr = []

        # read track vertices from file
        with open("shapes.txt", "r") as f:
            reader = csv.reader(f, delimiter=",")
            shapes = list(reader)
            shapes_arr = [list(map(ast.literal_eval, shape)) for shape in shapes]

        static_lines = []

        for shape in shapes_arr:
            for i in range(len(shape) - 1):
                static_lines.append(pymunk.Segment(static_body, shape[i], shape[i + 1], 0.0))

        for line in static_lines:
            line.elasticity = 0
            line.friction = 1

        print(len(static_lines))

        self._space.add(*static_lines)

        sensor_bodies = self._space.static_body
        shape_sensor = pymunk.Segment(sensor_bodies, (600, 800), (400, 600), 0.0)
        shape_sensor.sensor = True
        shape_sensor.collision_type = 1

        shape_sensor2 = pymunk.Segment(sensor_bodies, (800, 1000), (500, 600), 0.0)
        shape_sensor2.sensor = True
        shape_sensor2.collision_type = 2

        self._space.add(shape_sensor)
        self._space.add(shape_sensor2)

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                pass

    def _steer(self, value):
        if value == 0:
            return
        else:
            self._steerAngle += 0.1 * value

    def _accelerate(self, value):
        if value == -1:
            return
        else:
            self._playerBody.apply_impulse_at_local_point((0, 3 * value), (0, 0))

    def _break(self, value):
        if value == -1:
            return
        else:
            self._playerBody.apply_impulse_at_local_point((0, -4 * value), (0, 0))

    def _create_ball(self) -> None:
        mass = 1
        radius = 5
        inertia = pymunk.moment_for_circle(mass, 20, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = 500, 500
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0
        shape.friction = 1
        body.angle = math.pi
        shape.collision_type = 0

        self._space.add(body, shape)
        self._playerShape = shape
        self._playerBody = body

    def _clear_screen(self) -> None:
        self._screen.fill(pygame.Color("black"))

    def _draw_objects(self) -> None:
        self._space.debug_draw(self._draw_options)

    def start_callback(self, arbiter, space, data):
        print("Start lap")
        return True

    def end_callback(self, arbiter, space, data):
        print("End lap")
        return True


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
