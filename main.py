import math
import os

import numpy as np
import random
from typing import List
import pygame
import pymunk
import pymunk.pygame_util
import csv
import ast
import pygame_gui

window_width = 1500
window_length = 1000

ui_start_x = 1000

accelerate_image = pygame.image.load(os.path.join('resources', 'accelerate.png'))
not_accelerate_image = pygame.image.load(os.path.join('resources', 'not_accelerate.png'))
break_image = pygame.image.load(os.path.join('resources', 'break.png'))
not_break_image = pygame.image.load(os.path.join('resources', 'not_break.png'))


class Game(object):

    def __init__(self) -> None:
        # Space
        self.touch_track_counter = 0
        self._space = pymunk.Space()
        # self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._window_surface = pygame.display.set_mode((window_width, window_length))

        self._background = pygame.Surface((window_width, window_length))

        self._background.fill(pygame.Color("black"))

        self._clock = pygame.time.Clock()

        self._guiManager = pygame_gui.UIManager((window_width, window_length))
        self._guiManager.set_visual_debug_mode(True)

        gui_window = pygame_gui.elements.UIWindow(rect=pygame.rect.Rect((ui_start_x, 0), (500, 1000)),
                                                    window_display_title='window',
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


            time_delta = self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))



            # Get the state of each key
            keys = pygame.key.get_pressed()
            # key controls
            if keys[pygame.K_w]:
                self._accelerate(1)
            else:
                self._accelerate(-1)
            if keys[pygame.K_s]:
                self._break(1)
            else:
                self._break(-1)
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

            #updating events
            self._process_events()
            self._guiManager.update(time_delta)

            #resetting screen
            self._window_surface.blit(self._background, (0, 0))
            #self._window_surface.fill(pygame.Color("black"))

            #drawing
            self._draw_objects()
            self._guiManager.draw_ui(self._background)

            # observation
            print(self.observation())


            #timestep
            pygame.display.update()
            # pygame.display.flip()

    def _cast_rays(self, body, length):

        theta = body.angle + math.radians(90)
        fov = math.radians(90)

        count = 64
        draw_lines = True
        draw_contact = True

        # Define the angle increment for the rays
        angle_increment = fov / (count - 1)

        # Define the start angle for the rays
        start_angle = theta - fov / 2

        # pygame.draw.circle(self._window_surface, (0, 255, 0, 0.1), body.position, self._vision_radius, width=1)

        cone_start_x = length * math.cos(start_angle) + body.position.x
        cone_start_y = length * math.sin(start_angle) + body.position.y

        cone_end_x = length * math.cos(start_angle + fov) + body.position.x
        cone_end_y = length * math.sin(start_angle + fov) + body.position.y

        cone_rect = pygame.Rect(body.position.x - length, body.position.y - length, length * 2, length * 2)

        pygame.draw.line(self._window_surface, (0, 255, 0), body.position, (cone_start_x, cone_start_y), 1)
        pygame.draw.line(self._window_surface, (0, 255, 0), body.position, (cone_end_x, cone_end_y), 1)
        pygame.draw.arc(self._window_surface,
                        (0, 255, 0, 0.1),
                        cone_rect,
                        -(start_angle + fov),
                        -start_angle, width=1)

        # Create a list of angles for the segments
        angles = [i * math.pi / (count / 2) for i in range(count)]

        vision_contacts = []

        # Draw the rays
        for i in range(count):
            # Calculate the angle of the ray
            angle = start_angle + i * angle_increment
            # Calculate the end point of the ray
            end_x = length * math.cos(angle) + body.position.x
            end_y = length * math.sin(angle) + body.position.y
            end = (end_x, end_y)

            filter = pymunk.ShapeFilter(mask=0x1)

            # Perform a segment query against the space
            query = self._space.segment_query(body.position, end, 1, filter)

            query_res = [(np.linalg.norm(info.point - body.position), info.point) for info in query]

            if not len(query_res):
                break

            contact_point = min(query_res)
            if contact_point[0] > length - 2:
                contact_point = None

            if contact_point:
                vision_contacts.append(contact_point[1])

            # Draw a red dot at the point of intersection
            if contact_point:
                pygame.draw.circle(self._window_surface, (255, 0, 0), contact_point[1], 2)

            # Draw the segment
            if draw_lines and contact_point:
                pygame.draw.line(self._window_surface, (255, 255, 255), body.position, contact_point[1], 1)

        return vision_contacts

    def _add_static_scenery(self) -> None:
        static_body = self._space.static_body

        shapes_arr = []
        sectors_arr = []

        with open("shapes.txt", "r") as f:
            reader = csv.reader(f, delimiter=",")
            shapes = list(reader)
            shapes_arr = [list(map(ast.literal_eval, shape)) for shape in shapes]

        with open("sectors.txt", "r") as f:
            reader = csv.reader(f, delimiter=",")
            shapes = list(reader)
            sectors_arr = [list(map(ast.literal_eval, shape)) for shape in shapes]

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

        print(len(static_lines))

        self._space.add(*static_lines)

        # sectors
        sensor_bodies = self._space.static_body

        static_sector_lines = []

        for shape in sectors_arr:
            for i in range(len(shape) - 1):
                static_sector_lines.append(pymunk.Segment(sensor_bodies, shape[i], shape[i + 1], 0.0))

        for i in range(len(static_sector_lines)):
            static_sector_lines[i].elasticity = 0
            static_sector_lines[i].friction = 1
            static_sector_lines[i].sensor = True

        for i in range(1, len(static_sector_lines)):
            static_sector_lines[i].collision_type = i + 3
            static_sector_lines[i].filter = pymunk.ShapeFilter(categories=0x10)

        static_sector_lines[0].collision_type = 2
        static_sector_lines[0].filter = pymunk.ShapeFilter(categories=0x10)

        self._space.add(*static_sector_lines)

        # track collision
        track_col = self._space.add_collision_handler(0, 1)
        track_col.begin = self.track_callback_begin
        track_col.separate = self.track_callback_end
        track_col.pre_solve = self.track_callback_isTouch

        # start finish collision
        self._space.add_collision_handler(0, 2).begin = self.lap_callback

        # sectors collision
        for i in range(1, len(static_sector_lines)):
            col = self._space.add_collision_handler(0, i + 3)
            col.data["number"] = i + 1
            col.begin = self.sector_callback

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
            self.accelerate_ui.set_image(not_accelerate_image)
            return
        else:
            self.accelerate_ui.set_image(accelerate_image)
            self._playerBody.apply_impulse_at_local_point((0, 3 * value), (0, 0))

    def _break(self, value):
        if value == -1:
            self.break_ui.set_image(not_break_image)
            return
        else:
            self.break_ui.set_image(break_image)
            self._playerBody.apply_impulse_at_local_point((0, -4 * value), (0, 0))

    def _create_ball(self) -> None:
        mass = 1
        radius = 13
        inertia = pymunk.moment_for_circle(mass, 20, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = 130, 110
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0
        shape.friction = 1
        body.angle = math.pi * 3 / 2
        shape.collision_type = 0

        self._space.add(body, shape)
        self._playerShape = shape
        self._playerBody = body

    def _draw_objects(self) -> None:
        self._space.debug_draw(self._draw_options)

    def lap_callback(self, arbiter, space, data):
        print("set lap")

        return True

    def sector_callback(self, arbiter, space, data):
        print("set sector", data["number"])
        return True

    def track_callback_begin(self, arbiter, space, data):
        # print("exiting track")
        return True

    def track_callback_isTouch(self, arbiter, space, data):
        self.touch_track_counter += 1
        return True

    def track_callback_end(self, arbiter, space, data):
        print(self.touch_track_counter)
        self.touch_track_counter = 0
        return True

    def observation(self):
        obs = []


        # player position
        pos = self._playerBody.position
        # player vel TODO convert absolute vel to relative:
        vel = self._playerBody.velocity
        # player angle
        angl = [self._playerBody.angle, self._steerAngle]

        # todo points need to be relative
        points = self._cast_rays(self._playerBody, 600)


        # experimental
        # vibrations TODO
        # engine sounds TODO
        # g-force TODO


        return np.concatenate(
            [pos]
            +[vel]
            +[angl]
            +points
        )


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
