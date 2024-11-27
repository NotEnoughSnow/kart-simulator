import h5py


from kartSimulator.core.modules.launch_module import Launcher

class Play:


    def __init__(self, track_type, track_name, env_factory, save_config):


        env = env_factory.createEnv(track_type, track_name, "human")

        # Parameters for imitation learning
        # record_expert_data : to record data for imitation learning
        # expert_ep_count : number of episodes to record
        record_expert_data = False
        expert_ep_count = None
        expert_step_count = 100000
        player_name = "Amin"

        launcher = Launcher(env,
                            save_dir=save_config["save_dir"],
                            record=record_expert_data,
                            expert_ep_count=expert_ep_count,
                            player_name=player_name,
                            expert_steps_count=expert_step_count)

        launcher.launch(env)

