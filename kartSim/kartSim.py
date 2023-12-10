import math
import os
from typing import Union, Optional

import numpy as np
import pygame_gui
import pymunk
import pymunk.pygame_util
from gym import spaces

import core.env as core

import pygame

import kartSim.utils as Utils

window_width = 1500
window_length = 1000
VISION_COUNT = 64
VISION_LENGTH = 600

ui_start_x = 1000

accelerate_image = pygame.image.load(os.path.join('resources', 'accelerate.png'))
not_accelerate_image = pygame.image.load(os.path.join('resources', 'not_accelerate.png'))
break_image = pygame.image.load(os.path.join('resources', 'break.png'))
not_break_image = pygame.image.load(os.path.join('resources', 'not_break.png'))


class KartSim(core.Env):

    def __init__(self, render_mode):
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

        self._playerShape = None
        self._playerBody = None
        self._steerAngle = 0

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

        self._create_ball()
        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # steer, gas, brake

        # TODO use variables
        self.observation_space = spaces.Box(
            low=-400, high=400, shape=(VISION_COUNT * 2 + 1 + 2*2, 1), dtype=np.uint8
            # 2 points for vision ray count + position, angles, velocity
        )

        self.reward = 0.0

        self.render_mode = render_mode

    def step(self, action: Union[np.ndarray, int]):

        if action is not None:
            self._steer(-action[0])
            self._accelerate(action[1])
            self._break(action[2])

        # Progress time forward
        # TODO step based on FPS
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)
        # TODO lap and time counters

        time_delta = self._clock.tick(50)
        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

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

        # updating events
        self._process_events()
        self._guiManager.update(time_delta)

        # observation
        self.state = self.observation()

        step_reward = 0
        terminated = False
        truncated = False

        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1

            # TODO assign sector and lap based rewards

            # TODO if finish lap then truncated
            if False:
                truncated = True

            # TODO if collide with track then terminate
            if False:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset()
        # TODO physics and world reset code
        return self.step(None)[0], {}

    def render(self):
        # resetting screen
        self._window_surface.blit(self._background, (0, 0))
        # self._window_surface.fill(pygame.Color("black"))

        # drawing
        self._draw_objects()
        self._guiManager.draw_ui(self._background)

        pygame.display.update()
        # pygame.display.flip()

        # return self.isopen

    def close(self):
        if self._window_surface is not None:
            pygame.display.quit()
            # fixme isOpen variable
            pygame.quit()

    def _process_events(self) -> None:

        # enable player control
        self.playerInput()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._window_surface, "screenshots/karts.png")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                pass

    def playerInput(self):
        # Get the state of each key
        keys = pygame.key.get_pressed()
        # key controls
        if keys[pygame.K_w]:
            self._accelerate(1)
        else:
            self._accelerate(0)
        if keys[pygame.K_SPACE]:
            self._break(1)
        else:
            self._break(0)
        if keys[pygame.K_d] and self._steerAngle < 1:
            self._steer(1)
        if keys[pygame.K_a] and self._steerAngle > -1:
            self._steer(-1)

    def _cast_rays(self, body, length, ray_count):

        theta = body.angle + math.radians(90)
        fov = math.radians(90)

        draw_lines = True
        draw_contact = True
        draw_cone = False

        # Define the angle increment for the rays
        angle_increment = fov / (ray_count - 1)

        # Define the start angle for the rays
        start_angle = theta - fov / 2

        # pygame.draw.circle(self._window_surface, (0, 255, 0, 0.1), body.position, self._vision_radius, width=1)

        cone_start_x = length * math.cos(start_angle) + body.position.x
        cone_start_y = length * math.sin(start_angle) + body.position.y

        cone_end_x = length * math.cos(start_angle + fov) + body.position.x
        cone_end_y = length * math.sin(start_angle + fov) + body.position.y

        cone_rect = pygame.Rect(body.position.x - length, body.position.y - length, length * 2, length * 2)
        if draw_cone:
            pygame.draw.line(self._window_surface, (0, 255, 0), body.position, (cone_start_x, cone_start_y), 1)
            pygame.draw.line(self._window_surface, (0, 255, 0), body.position, (cone_end_x, cone_end_y), 1)
            pygame.draw.arc(self._window_surface,
                            (0, 255, 0, 0.1),
                            cone_rect,
                            -(start_angle + fov),
                            -start_angle, width=1)

        # Create a list of angles for the segments
        angles = [i * math.pi / (ray_count / 2) for i in range(ray_count)]

        vision_contacts = []

        # Draw the rays
        for i in range(ray_count):
            # Calculate the angle of the ray
            angle = start_angle + i * angle_increment
            # Calculate the end point of the ray
            end_x = length * math.cos(angle) + body.position.x
            end_y = length * math.sin(angle) + body.position.y
            end = (end_x, end_y)

            filter = pymunk.ShapeFilter(mask=0x1)

            # Perform a segment query against the space
            query = self._space.segment_query(body.position, end, 1, filter)

            query_res = [[np.linalg.norm(info.point - body.position), info.point] for info in query]

            if len(query_res) == 0:
                query_res.append((length, (0, 0)))

            contact_point = min(query_res)

            if contact_point[0] > length - 2:
                contact_point = (0, (0, 0))
                vision_contacts.append([0, 0])
            else:
                vision_contacts.append(contact_point[1] - self._playerBody.position)

            # Draw a red dot at the point of intersection
            if draw_contact and contact_point[0] != 0:
                pygame.draw.circle(self._window_surface, (255, 0, 0), contact_point[1], 2)

            # Draw the segment
            if draw_lines and contact_point[0] != 0:
                pygame.draw.line(self._window_surface, (255, 255, 255), body.position, contact_point[1], 1)

        return vision_contacts

    def _add_static_scenery(self) -> None:
        static_body = self._space.static_body

        shapes_arr = Utils.readTrackFile("shapes.txt")
        sectors_arr = Utils.readTrackFile("sectors.txt")

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

    def _steer(self, value):
        """steering control

        :param value: (-1..1)
        :return:
        """
        if value == 0:
            return
        else:
            self._steerAngle += 0.1 * value

    def _accelerate(self, value):
        """acceleration control

        :param value: (0..1)
        :return:
        """
        if value == 0:
            self.accelerate_ui.set_image(not_accelerate_image)
            return
        else:
            self.accelerate_ui.set_image(accelerate_image)
            self._playerBody.apply_impulse_at_local_point((0, 3 * value), (0, 0))

    def _break(self, value):
        """breaking control

        :param value: (0..1)
        :return:
        """
        if value == 0:
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
        # print(self.touch_track_counter)
        self.touch_track_counter = 0
        return True

    def observation(self):
        obs = []

        # player position
        pos = self._playerBody.position
        # player vel TODO convert absolute vel to relative:
        vel = self._playerBody.velocity.__abs__()
        # player angle
        angl = [self._playerBody.angle, self._steerAngle]

        # todo account for missing points
        points = self._cast_rays(self._playerBody, VISION_LENGTH, VISION_COUNT)

        # experimental
        # vibrations TODO
        # engine sounds TODO
        # g-force TODO

        return np.concatenate(
            [pos]
            + [[vel]]
            + [angl]
            + points
        )
