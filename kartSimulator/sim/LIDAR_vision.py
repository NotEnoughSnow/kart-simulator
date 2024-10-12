import pygame
import math
import pymunk
import numpy as np
import torch

#VISION_LENGTH = 300
VISION_LENGTH = 100
NO_VISION_CONSTANT = VISION_LENGTH / 6
VISION_FOV = 360
RAY_COUNT = 60

halfwinsize = 5
#torch.manual_seed(0)

conv_layer = torch.nn.Conv1d(in_channels=1,
                             out_channels=1,
                             kernel_size=(2 * halfwinsize + 1),
                             padding='valid')

max_input = VISION_LENGTH + NO_VISION_CONSTANT

maximum = (conv_layer.weight.data.clamp(min=0).sum() * max_input + conv_layer.bias.data).item()
minimum = (conv_layer.weight.data.clamp(max=0).sum() * max_input + conv_layer.bias.data).item()


def apply_convolution(wraparound_data):

    convolved_data = conv_layer(wraparound_data.unsqueeze(0).unsqueeze(0))

    return convolved_data.squeeze().squeeze().detach().numpy()

def apply_circularity(data):
    data = torch.tensor(data)

    wraparound_data = torch.cat(
        [data[-halfwinsize:], data, data[:halfwinsize]]).float()

    return wraparound_data

'''
def draw_rays(window_surface, anchor, contact_point, scalar, draw_contact, draw_lines):
    for point in contact_point:
        point = point[0] * scalar, point[1] * scalar

        # Draw a red dot at the point of intersection
        if draw_contact and contact_point != (0, 0):
            pygame.draw.circle(window_surface, (255, 0, 0), point + anchor, 2)

        # Draw the segment
        if draw_lines and contact_point != (0, 0):
            pygame.draw.line(window_surface, (255, 255, 255), anchor, point + anchor, 1)
'''
def draw_rays(window_surface, anchor, contact_point, scalar, draw_contact, draw_lines):

    for i in range(len(contact_point)):
        point = contact_point[i][0] * scalar, contact_point[i][1] * scalar

        # Draw a red dot at the point of intersection
        if draw_contact and contact_point != (0, 0):
            if i == len(contact_point)//2 + 1:
                pygame.draw.circle(window_surface, (0, 255, 0), point + anchor, 2)
            else:
                pygame.draw.circle(window_surface, (255, 0, 0), point + anchor, 2)

        # Draw the segment
        if draw_lines and contact_point != (0, 0):
            if i == len(contact_point)//2 + 1:
                pygame.draw.line(window_surface, (0, 255, 0), anchor, point + anchor, 1)
            else:
                pygame.draw.line(window_surface, (255, 255, 255), anchor, point + anchor, 1)


def draw_cone(window_surface, body):

    theta = body.angle + math.radians(90)
    fov = math.radians(VISION_FOV)

    # Define the angle increment for the rays
    angle_increment = fov / (RAY_COUNT - 1)

    # Define the start angle for the rays
    start_angle = theta - fov / 2

    # pygame.draw.circle(self._window_surface, (0, 255, 0, 0.1), body.position, self._vision_radius, width=1)

    cone_start_x = VISION_LENGTH * math.cos(start_angle) + body.position.x
    cone_start_y = VISION_LENGTH * math.sin(start_angle) + body.position.y

    cone_end_x = VISION_LENGTH * math.cos(start_angle + fov) + body.position.x
    cone_end_y = VISION_LENGTH * math.sin(start_angle + fov) + body.position.y

    cone_rect = pygame.Rect(body.position.x - VISION_LENGTH, body.position.y - VISION_LENGTH, VISION_LENGTH * 2, VISION_LENGTH * 2)

    pygame.draw.line(window_surface, (0, 255, 0), body.position, (cone_start_x, cone_start_y), 1)
    pygame.draw.line(window_surface, (0, 255, 0), body.position, (cone_end_x, cone_end_y), 1)
    pygame.draw.arc(window_surface,
                    (0, 255, 0, 0.1),
                    cone_rect,
                    -(start_angle + fov),
                    -start_angle, width=1)

def cast_rays(space, body):
    theta = body.angle + math.radians(90)
    fov = math.radians(VISION_FOV)

    # Define the angle increment for the rays
    angle_increment = fov / (RAY_COUNT - 1)

    # Define the start angle for the rays
    start_angle = theta - fov / 2

    # Create a list oaf angles for the segments
    angles = [i * math.pi / (RAY_COUNT / 2) for i in range(RAY_COUNT)]

    vision_contacts = []

    # Draw the rays
    for i in range(RAY_COUNT-1):
        # Calculate the angle of the ray
        angle = start_angle + i * angle_increment
        # Calculate the end point of the ray
        end_x = VISION_LENGTH * math.cos(angle) + body.position.x
        end_y = VISION_LENGTH * math.sin(angle) + body.position.y
        end = (end_x, end_y)

        filter = pymunk.ShapeFilter(mask=0x1)

        # Perform a segment query against the space
        query = space.segment_query(body.position, end, 1, filter)

        query_res = [[np.linalg.norm(info.point - body.position), info.point] for info in query]

        if len(query_res) == 0:
            query_res.append((VISION_LENGTH, (0, 0)))

        contact_point = min(query_res)

        if contact_point[0] > VISION_LENGTH - 2:
            contact_point = (0, (0, 0))
            vision_contacts.append([0, 0])
        else:
            vision_contacts.append(contact_point[1] - body.position)

    return vision_contacts


def cast_rays_lengths(space, body):
    theta = body.angle + math.radians(90)
    fov = math.radians(VISION_FOV)

    # Define the angle increment for the rays
    angle_increment = fov / (RAY_COUNT - 1)

    # Define the start angle for the rays
    start_angle = theta - fov / 2

    # Create a list oaf angles for the segments
    angles = [i * math.pi / (RAY_COUNT / 2) for i in range(RAY_COUNT)]

    vision_contacts = []
    vision_lengths = []

    # Draw the rays
    for i in range(RAY_COUNT-1):
        # Calculate the angle of the ray
        angle = start_angle + i * angle_increment
        # Calculate the end point of the ray
        end_x = VISION_LENGTH * math.cos(angle) + body.position.x
        end_y = VISION_LENGTH * math.sin(angle) + body.position.y
        end = (end_x, end_y)

        filter = pymunk.ShapeFilter(mask=0x1)

        # Perform a segment query against the space
        query = space.segment_query(body.position, end, 1, filter)

        query_res = [[np.linalg.norm(info.point - body.position), info.point] for info in query]

        if len(query_res) == 0:
            query_res.append((VISION_LENGTH, (0, 0)))

        contact_point = min(query_res)

        if contact_point[0] > VISION_LENGTH - 2:
            contact_point = (0, (0, 0))
            vision_contacts.append([0, 0])
            vision_lengths.append(VISION_LENGTH + NO_VISION_CONSTANT)
        else:
            vision_contacts.append(contact_point[1] - body.position)
            vision_lengths.append(contact_point[0])

    return vision_contacts, vision_lengths
