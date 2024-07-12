import pymunk

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

WORLD_CENTER = [500, 500]


class Agent:
    def __init__(self, space, color, mode):
        self.vars = {}

        mass = BOT_WEIGHT
        self.radius = BOT_SIZE * PPM
        inertia = pymunk.moment_for_circle(mass, 0, self.radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = [0, 0]
        shape = pymunk.Circle(body, self.radius, (0, 0))
        shape.elasticity = 0
        shape.friction = 1
        shape.collision_type = 0

        shape.filter = pymunk.ShapeFilter(categories=0b001, mask=0b110)

        space.add(body, shape)
        self.playerShape = shape
        self.playerBody = body

        self.color = color

    def init_vars(self, position):
        self.playerBody.position = position
        self.playerBody.velocity = 0, 0

        self.vars = {"break_value": 0,
                     "accel_value": 0,
                     "steer_right_value": 0,
                     "steer_left_value": 0,
                     "reward": 0,
                     "prev_reward": 0,
                     "position": self.playerBody.position,
                     "velocity": 0,
                     }
