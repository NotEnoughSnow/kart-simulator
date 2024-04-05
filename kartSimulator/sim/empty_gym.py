import math
import os
from typing import Union, Optional

import numpy as np
import pygame_gui
import pymunk
import pymunk.pygame_util
from gym import spaces
from pymunk import Vec2d

import kartSimulator.core.env as core
import kartSimulator.sim.utils as Utils

import pygame

PPM = 100

BOT_SIZE = 0.192
BOT_WEIGHT = 1

# 1 meter in 4.546 sec or 0.22 meters in 1 sec
# at 0.22 m/s, the bot should walk 1 meter in 4.546 seconds
MAX_VELOCITY = 0.22 * PPM

# 0.1 rad/frames, 2pi in 1.281 sec

# example : 0.0379 rad/frames, for frames = 48
# or        1.82 rad/sec
RAD_VELOCITY = 2.84

window_width = 1500
window_length = 1000
VISION_COUNT = 64
VISION_LENGTH = 600
VISION_FOV = 90

ui_start_x = 1000

accelerate_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'accelerate.png'))
not_accelerate_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'not_accelerate.png'))
break_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'break.png'))
not_break_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'not_break.png'))


class KartSim(core.Env):

    def __init__(self, render_mode=None, manual=False, train=False):

        # responsible for assigning control to player
        self.manual = manual
        self.render_mode = render_mode

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
            low=-1, high=1, shape=(5,), dtype=np.uint8
        )
        # do nothing, left, right, gas, brake

        # TODO use variables
        self.observation_space = spaces.Box(
            low=-400, high=400, shape=(1 + 2 * 2, ), dtype=np.uint8
            # 2 points for vision ray count + position, angles, velocity
        )

        self.initial_pos = 300, 450
        self.initial_angle = math.pi * 3 / 2
        self.vision_points = []
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
            #self.FPS = self._clock.get_fps()
            #print("what ", self._clock.get_fps())

        break_value = 0
        accel_value = 0
        steer_right_value = 0
        steer_left_value = 0

        print("action", action)

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
        state = self.observation()

        step_reward = 0
        terminated = False
        truncated = False

        # FIXME should look better
        next_target_distance = (self._playerBody.position - self.sector_info[self.next_sector_name][1]).__abs__()
        self.next_target_distance = np.exp(4-(next_target_distance/200))-20

        if action is not None or self.manual:  # First step without action, called from reset()
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


        #print( self._playerBody.position)

        #(shape[0][0] + shape[1][0]) / 2

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

        return self.step(None)[0], {}

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

        # self._draw_cone(self._playerBody, VISION_LENGTH, VISION_COUNT, VISION_FOV)
        self._draw_rays(Vec2d(ui_start_x + 250, 850), self.vision_points, 0.3, True, True)
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

        if self.manual:
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

        break_value = 0
        accel_value = 0
        steer_value = 0

        # Get the state of each key
        keys = pygame.key.get_pressed()
        # key controls
        if keys[pygame.K_w]:
            accel_value = self._accelerate(1)
        else:
            accel_value = self._accelerate(-1)
        if keys[pygame.K_SPACE]:
            break_value = self._break(1)
        else:
            break_value = self._break(-1)
        if keys[pygame.K_d]:
            steer_value = self._steer_right(1)
        if keys[pygame.K_a]:
            steer_value = self._steer_left(1)

        self.break_value = break_value
        self.accel_value = accel_value
        self.steer_value = steer_value

    def _draw_cone(self, body, length, ray_count, fov):

        theta = body.angle + math.radians(90)
        fov = math.radians(fov)

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

        pygame.draw.line(self._window_surface, (0, 255, 0), body.position, (cone_start_x, cone_start_y), 1)
        pygame.draw.line(self._window_surface, (0, 255, 0), body.position, (cone_end_x, cone_end_y), 1)
        pygame.draw.arc(self._window_surface,
                        (0, 255, 0, 0.1),
                        cone_rect,
                        -(start_angle + fov),
                        -start_angle, width=1)

    def _draw_rays(self, anchor, contact_point, scalar, draw_contact, draw_lines):

        for point in contact_point:
            point = point[0] * scalar, point[1] * scalar

            # Draw a red dot at the point of intersection
            if draw_contact and contact_point != (0, 0):
                pygame.draw.circle(self._window_surface, (255, 0, 0), point + anchor, 2)

            # Draw the segment
            if draw_lines and contact_point != (0, 0):
                pygame.draw.line(self._window_surface, (255, 255, 255), anchor, point + anchor, 1)

    def _cast_rays(self, body, length, ray_count, fov):

        theta = body.angle + math.radians(90)
        fov = math.radians(fov)

        # Define the angle increment for the rays
        angle_increment = fov / (ray_count - 1)

        # Define the start angle for the rays
        start_angle = theta - fov / 2

        # Create a list oaf angles for the segments
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

        return vision_contacts

    def _add_static_scenery(self) -> None:
        static_body = self._space.static_body

        shapes_arr = Utils.readTrackFile("kartSimulator\\sim\\resources\\boxes.txt")
        sectors_arr = Utils.readTrackFile("kartSimulator\\sim\\resources\sectors_box.txt")

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
            sector_midpoints.append([(shape[0][0] + shape[1][0])/2,(shape[0][1] + shape[1][1])/2])

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
            self.sector_info["sector " + str(i)].append(sector_midpoints[i-1])


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
                self._playerBody.apply_impulse_at_local_point((0, 4 * value), (0, 0))
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
                self._playerBody.apply_impulse_at_local_point((0, -4 * value), (0, 0))
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
            #print("visited " + name + " for the first time")
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

    def observation(self):
        obs = []

        # player position
        self.position = self._playerBody.position
        # player vel TODO convert absolute vel to relative:
        self.velocity = self._playerBody.velocity.__abs__()
        # player angle
        angl = [self._playerBody.angle, self._steerAngle]

        # todo account for missing points
        self.vision_points = self._cast_rays(self._playerBody, VISION_LENGTH, VISION_COUNT, VISION_FOV)

        # experimental
        # vibrations TODO
        # engine sounds TODO
        # g-force TODO

        return np.concatenate(
            [self.position]
            + [[self.velocity]]
            + [angl]
        )
