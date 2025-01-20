from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import dexhub 
from tqdm import tqdm 
import numpy as np 
import os 
from mrl_utils.mj_utils import get_sim
import mujoco


def get_sim_and_dataset(dataset_uuid, dataset_name = None, num_next_actions = 1): 

    data_dir = dexhub.download_dataset(uuid=dataset_uuid, load_dir = "./", load_to_mem = False)
    dex_data_dir = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dex')]

    mujoco_model = get_sim(dex_data_dir)
    mujoco_data = mujoco.MjData(mujoco_model)

    renderer = mujoco.Renderer(mujoco_model, 128, 128) 

    data = dexhub.load(dex_data_dir[0])
    obs_dim = data.data[0].obs.mj_qpos.shape
    act_dim = data.data[0].act.mj_ctrl.shape

    dim_info = {
        'obs_state': obs_dim,
        'obs_image': (3, 128, 128),
        'action': act_dim
    }

    robot_cfg = init_hydra_config("./panda.yaml")
    robot = make_robot(robot_cfg)

    # create random name for the dataset  (if not provided)
    # style:  xxxx/xxxx 
    if dataset_name is None:
        dataset_name = "younghyopark/" + str(np.random.randint(0, 1000))

    try: 
        dataset = LeRobotDataset.create(dataset_name, fps = 10, robot = robot)
    except: 
        # add random index to the dataset_name
        dataset_name = dataset_name + str(np.random.randint(0, 1000))
        dataset = LeRobotDataset.create(dataset_name, fps = 10, robot = robot)

    dataset.features['observation.state']['shape'] = obs_dim
    dataset.features['action']['shape'] = act_dim

    # Load the dataset
    for d in dex_data_dir:
        data = dexhub.load(d)

        # Convert the dataset
        for i in range(len(data.data)):
            obs = data.data[i].obs.mj_qpos
            act = data.data[i].act.mj_ctrl

            mujoco_data.qpos = obs 
            mujoco.mj_step(mujoco_model, mujoco_data) 

            renderer.update_scene(mujoco_data)
            pixels = renderer.render()

            dataset.add_frame({
                'observation.state': obs,
                # 'observaton.environment_state': obs, 
                'observation.images.main': pixels, 
                'action': act
            })
        # Now try saving
        dataset.save_episode(task="test", encode_videos=False)

    dataset.consolidate()

    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.images.main": [0.0],
        "observation.state": [0.0],
        # create a list of actions to be loaded
        # from 0.0, load num_next_actions actions with 0.1 second interval
        "action": [0.0] + [0.1 * i for i in range(1, num_next_actions)]
    }



    final_dataset = LeRobotDataset(dataset_name, delta_timestamps = delta_timestamps, local_files_only=True)

    return mujoco_model, final_dataset, dim_info
