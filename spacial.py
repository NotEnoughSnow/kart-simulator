import math
import numpy as np
import random
from typing import List
import pygame
import pymunk
import pymunk.pygame_util
import csv
import ast
import pygame_gui


class Game(object):

    def __init__(self) -> None:

        self.width = 1000
        self.height = 1000

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
        self._window_surface = pygame.display.set_mode((self.width, self.height))

        self._background = pygame.Surface((self.width, self.height))
        self._background.fill(pygame.Color("black"))

        self._clock = pygame.time.Clock()

        self._guiManager = pygame_gui.UIManager((1000, 1000))

        # hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((350, 275), (100, 50)),
        #                                            text='UI',
        #                                            manager=self._guiManager)

        self._draw_options = pymunk.pygame_util.DrawOptions(self._window_surface)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        self._playerShape = None
        self._playerBody = None
        self._steerAngle = 0
        self._vision_radius = 300

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

    def run(self) -> None:

        self._create_ball()
        self.observation()

        # Main loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            time_delta = self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

            mv_value = 3

            # Get the state of each key
            keys = pygame.key.get_pressed()
            # key controls
            if keys[pygame.K_w]:
                self._playerBody.apply_impulse_at_local_point((0, -mv_value), (0, 0))
            if keys[pygame.K_s]:
                self._playerBody.apply_impulse_at_local_point((0, mv_value), (0, 0))
            if keys[pygame.K_d]:
                self._playerBody.angle += 0.1
            if keys[pygame.K_a]:
                self._playerBody.angle -= 0.1


            # updating events
            self._process_events()
            self._guiManager.update(time_delta)

            # resetting screen
            self._window_surface.fill(pygame.Color("black"))





            # drawing
            self._draw_objects()
            self._guiManager.draw_ui(self._background)
            self._cast_rays_circular(self._playerBody, self._vision_radius)


            # timestep
            #pygame.display.update()
            pygame.display.flip()

    def _generate_box(self, body, side, gap):

        gap_arr = [0, 0, 0, 0, 0, 0, 0, gap]


        shapes = {
            pymunk.Segment(body,
                           (self.width / 2 - side + gap_arr[0], self.height / 2 - side),
                           (self.width / 2 + side - gap_arr[1], self.height / 2 - side), 0.0),
            pymunk.Segment(body,
                           (self.width / 2 + side, self.height / 2 - side + gap_arr[2]),
                           (self.width / 2 + side, self.height / 2 + side - gap_arr[3]), 0.0),
            pymunk.Segment(body,
                           (self.width / 2 + side - gap_arr[4], self.height / 2 + side),
                           (self.width / 2 - side + gap_arr[5], self.height / 2 + side), 0.0),
            pymunk.Segment(body,
                           (self.width / 2 - side, self.height / 2 + side - gap_arr[6]),
                           (self.width / 2 - side, self.height / 2 - side + gap_arr[7]), 0.0),
        }
        return shapes


    def _add_static_scenery(self) -> None:
        static_body = self._space.static_body

        inner_box_shapes = []
        outer_box_shapes = []

        inner_lines = []
        outer_lines = []

        inner_box_value = 100
        outer_box_value = 200

        inner_box_shapes = self._generate_box(static_body, inner_box_value, 75)
        outer_box_shapes = self._generate_box(static_body, outer_box_value, 0)

        for shape in inner_box_shapes:
            inner_lines.append(shape)

        for shape in outer_box_shapes:
            outer_lines.append(shape)

        for line in inner_lines:
            line.sensor = True
            line.collision_type = 1

        for line in outer_lines:
            line.sensor = True
            line.collision_type = 2

        self._space.add(*inner_lines)
        self._space.add(*outer_lines)

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

    def _create_vision(self, body):
        count = 16
        length = 400


        # Create a list of angles for the segments
        angles = [i * math.pi / (count/2) for i in range(count)]

        # Create a list of segments
        segments = []
        for i in range(count):
            # Calculate the start and end points of the segment
            start_x = length * math.cos(angles[i])
            start_y = length * math.sin(angles[i])
            end_x = length * math.cos(angles[(i + 1) % count])
            end_y = length * math.sin(angles[(i + 1) % count])
            start = (start_x, start_y)
            end = (end_x, end_y)
            # Create the segment
            segment = pymunk.Segment(body, start, end, 1)
            # Set the properties of the segment
            segment.elasticity = 0.5
            segment.friction = 0.5
            # Add the segment to the list
            segments.append(segment)

        return segments

    def _cast_rays(self, body, length):

        theta = body.angle+math.radians(-90)
        fov = math.radians(90)

        count = 64
        draw_lines = True
        draw_contact = True

        # Define the angle increment for the rays
        angle_increment = fov / (count - 1)

        # Define the start angle for the rays
        start_angle = theta - fov / 2

        #pygame.draw.circle(self._window_surface, (0, 255, 0, 0.1), body.position, self._vision_radius, width=1)

        cone_start_x = length * math.cos(start_angle) + body.position.x
        cone_start_y = length * math.sin(start_angle) + body.position.y

        cone_end_x = length * math.cos(start_angle+fov) + body.position.x
        cone_end_y = length * math.sin(start_angle+fov) + body.position.y

        cone_rect = pygame.Rect(body.position.x-length, body.position.y-length, length*2, length*2)

        pygame.draw.line(self._window_surface, (0, 255, 0), body.position, (cone_start_x, cone_start_y), 1)
        pygame.draw.line(self._window_surface, (0, 255, 0), body.position, (cone_end_x, cone_end_y), 1)
        pygame.draw.arc(self._window_surface,
                        (0, 255, 0, 0.1),
                        cone_rect,
                        -(start_angle+fov),
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

            # Perform a segment query against the space
            query = self._space.segment_query(body.position, end, 1, pymunk.ShapeFilter())

            # TODO lines that don't touch shouldn't generate contact points

            query_res = [(np.linalg.norm(info.point - body.position), info.point) for info in query ]
            contact_point = min(query_res)
            if contact_point[0] > length-2:
                contact_point = None

            vision_contacts.append(contact_point)
            
            # Draw a red dot at the point of intersection
            if contact_point:
                pygame.draw.circle(self._window_surface, (255, 0, 0), contact_point[1], 2)

            # Draw the segment
            if draw_lines and contact_point:
                pygame.draw.line(self._window_surface, (255, 255, 255), body.position, contact_point[1], 1)

        return vision_contacts

    def _cast_rays_circular(self, body, length):

        count = 64
        draw_lines = True
        draw_contact = True


        pygame.draw.circle(self._window_surface, (0, 255, 0, 0.1), body.position, length, width=1)

        cone_rect = pygame.Rect(body.position.x - length, body.position.y - length, length * 2, length * 2)

        # Create a list of angles for the segments
        angles = [i * math.pi / (count/2) for i in range(count)]

        vision_contacts = []

        # Draw the rays
        for i in range(count):
            # Calculate the end point of the ray
            end_x = length * math.cos(angles[i]) + body.position.x
            end_y = length * math.sin(angles[i]) + body.position.y
            end = (end_x, end_y)

            # Perform a segment query against the space
            query = self._space.segment_query(body.position, end, 1, pymunk.ShapeFilter())

            # TODO lines that don't touch shouldn't generate contact points

            query_res = [(np.linalg.norm(info.point - body.position), info.point) for info in query]
            contact_point = min(query_res)
            print(query_res)

            if contact_point[0] > length - 2:
                contact_point = None

            vision_contacts.append(contact_point)

            # Draw a red dot at the point of intersection
            if contact_point:
                pygame.draw.circle(self._window_surface, (255, 0, 0), contact_point[1], 2)

            # Draw the segment
            if draw_lines and contact_point:
                pygame.draw.line(self._window_surface, (255, 255, 255), body.position, contact_point[1], 1)

        return vision_contacts

    def _create_ball(self) -> None:
        mass = 1
        radius = 13
        inertia = pymunk.moment_for_circle(mass, 20, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = 50, 50
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0
        shape.friction = 1
        shape.collision_type = 0

        #vision_cone = self._create_vision(body)

        self._space.add(body, shape)

        #self._space.add(*vision_cone)

        self._playerShape = shape
        self._playerBody = body


    def _draw_objects(self) -> None:
        self._space.debug_draw(self._draw_options)

    def observation(self):
        obs = []

        # player position
        obs.append(self._playerBody.position[0])
        obs.append(self._playerBody.position[1])
        # player vel TODO convert absolute vel to relative:
        obs.append(self._playerBody.velocity[0])
        obs.append(self._playerBody.velocity[1])
        # player angle
        obs.append(self._playerBody.angle)
        obs.append(self._steerAngle)
        # vision TODO

        # experimental
        # vibrations TODO
        # engine sounds TODO
        # g-force TODO

        return obs


def main():
    game = Game()
    game.run()


main()
