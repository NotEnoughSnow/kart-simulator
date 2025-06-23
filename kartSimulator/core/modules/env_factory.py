
from kartSimulator.sim.maps.track_factory import TrackFactory
import kartSimulator.sim.observation_types as obs_types
import kartSimulator.sim.grid_env as simple_env
import kartSimulator.sim.steer_env as steer_env
import kartSimulator.sim.steer_gazebo as steer_gazebo

from kartSimulator.sim import calibrate_new_2


class EnvFactory:

    def __init__(self, env_name):

        self.env_name = env_name

        self.track_list = {
            "small_S": {"track_file": "boxes.txt",
                      "sectors_file": "sectors_box.txt",
                      "initial_pos": [330, 450],
                      "rand goal": None, },
            "big_S": {"track_file": "big_S_track.txt",
                      "sectors_file": "big_S_sectors.txt",
                      "initial_pos": [150, 200],
                      "rand goal": None, },
            "full_track": {"track_file": "shapes.txt",
                      "sectors_file": "sectors.txt",
                      "initial_pos": [180, 100],
                      "rand goal": None, },
            "map_1": {"track_file": "curr_maps/map_1/track.txt",
                      "sectors_file": "curr_maps/map_1/sectors.txt",
                      "initial_pos": [170, 550],
                      "rand goal": None, },
            "map_2": {"track_file": "curr_maps/map_2/track.txt",
                      "sectors_file": "curr_maps/map_2/sectors.txt",
                      "initial_pos": [200, 450],
                      "rand goal": ([0, 0], [-500, 500])},
            "map_3": {"track_file": "curr_maps/map_3/track.txt",
                      "sectors_file": "curr_maps/map_3/sectors.txt",
                      "initial_pos": [190, 390],
                      "rand goal": None, },
            "map_4": {"track_file": "curr_maps/map_4/track.txt",
                      "sectors_file": "curr_maps/map_4/sectors.txt",
                      "initial_pos": [190, 250],
                      "rand goal": None, },

        }

        self.obs_steer = [obs_types.LIDAR,
               obs_types.POSITION,
               obs_types.VELOCITY,
               obs_types.ROTATION,
               ]

        self.obs_grid = [obs_types.LIDAR,
                         obs_types.POSITION,
                         obs_types.VELOCITY,
                         ]

        grid_env_player_args = {
            "player_acc_rate": 15,
            "max_velocity": 2,
            "bot_size": 0.192,
            "bot_weight": 1,
            "vision_length": 6,
            "vision_fov": 360,
            "vision_ray_count": 60,
        }
        #max_vel = 0.22
        max_vel = 3.5
        #rad_vel = 2.84
        rad_vel = 0.86 * 2.84
        steer_env_player_args = {
            "player_acc_rate": max_vel / 0.66, # arbitrary
            "player_break_rate": max_vel / 0.5, # arbitrary
            #"max_velocity": 4, # based on max bot speed, multiplied for convenience
            "max_velocity": max_vel, # based on max bot speed, multiplied for convenience
            #"rad_velocity": 5 * 2.84, # based on max bot rad speed, multiplied for convenience
            "rad_velocity": rad_vel, # based on max bot rad speed, multiplied for convenience

            "bot_size": 0.192,
            "bot_weight": 1,
            "vision_length": 3.5,
            "vision_fov": 360,
            "vision_ray_count": 60,
        }

        steer_gazebo_player_args = {
            "player_acc_rate": 40 * 0.22 / 2 , # based on max bot speed
            "rad_velocity": 2.84 * 2, # based on max bot rad speed

            "bot_size": 0.192,
            "bot_weight": 1,
            "vision_length": 3.5,
            "vision_fov": 360,
            "vision_ray_count": 60,
        }


        '''        simple_env_player_args = {
            "player_acc_rate": 15,
            "max_velocity": 2,
            "bot_size": 0.192,
            "bot_weight": 1,
        }
        base_env_player_args = {
            "player_acc_rate": 5,
            "player_break_rate": 5,
            "max_velocity": 2,
            "rad_velocity": 3 * 2.84,
            "bot_size": 0.192,
            "bot_weight": 1,
        }'''

        self.track_args = {
            # "track_file": "shapes.txt",
            # "sectors_file": "sectors.txt",
            "track_file": "boxes.txt",
            "sectors_file": "sectors_box.txt",
            "rand goal": None,

            "corridor_size": 50,

            "spawn_range": 400,
            "fixed_goal": [200, -200],

            "initial_pos": [330, 450]
            # "initial_pos": [180, 100]
        }

        self.rew_adj_grid = {
            "passive": 0,
            "dist": 0,
            "act_dist": 1,
            "sector_time": 1,
        }

        self.rew_adj_steer = {
            "passive": 0,
            "dist": 0,
            "act_dist": 0.5,
            "sector_time": 1,
            "steer": 0.7,
        }

        self.rew_adj_gazebo = {
            "passive": 0,
            "dist": 0,
            "act_dist": 0.2,
            "sector_time": 2,
            "steer": 0.7,
            "finish": 500,
            "wall": 1,
        }

        player_args = None
        rew_adj = None
        obs_seq = None

        if env_name == simple_env:
            player_args = grid_env_player_args
            rew_adj = self.rew_adj_grid
            obs_seq = self.obs_grid
        elif env_name == steer_env:
            player_args = steer_env_player_args
            rew_adj = self.rew_adj_steer
            obs_seq = self.obs_steer
        elif env_name == steer_gazebo:
            player_args = steer_gazebo_player_args
            rew_adj = self.rew_adj_gazebo
            obs_seq = self.obs_steer
        elif env_name == calibrate_new_2:
            player_args = steer_gazebo_player_args
            rew_adj = self.rew_adj_gazebo
            obs_seq = self.obs_steer


        self.env_args = {
            "obs_seq": obs_seq,
            "reset_time": 1500,
            "track": None,
            "player_args": player_args,
            "rew_adj": rew_adj,
        }

    def set_rew_adj(self, rew_adj):
        self.rew_adj = rew_adj
        self.env_args["rew_adj"] = self.rew_adj

    def set_obs(self, obs):
        self.self.obs = obs
        self.env_args["obs_seq"] = self.obs


    def createEnv(self, track_type, track_name, render):

        self.track_args.update({
            "track_file": self.track_list[track_name]["track_file"],
            "sectors_file": self.track_list[track_name]["sectors_file"],
            "initial_pos": self.track_list[track_name]["initial_pos"],
            "rand goal": self.track_list[track_name]["rand goal"],
        })


        track = TrackFactory.create_track(track_type, **self.track_args)

        self.env_args["track"] = track


        env = self.env_name.KartSim(render_mode=render, train=True, **self.env_args)

        return env

    def updateFactory(self, track_type, track_name):

        self.track_args.update({
            "track_file": self.track_list[track_name]["track_file"],
            "sectors_file": self.track_list[track_name]["sectors_file"],
            "initial_pos": self.track_list[track_name]["initial_pos"],
            "rand goal": self.track_list[track_name]["rand goal"],
        })


        track = TrackFactory.create_track(track_type, **self.track_args)

        self.env_args["track"] = track



