from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import dexhub 
from tqdm import tqdm 
import numpy as np 
import os 
from mrl_utils.mj_utils import get_sim
import mujoco


def get_sim_and_dataset(dataset_uuid, dataset_name = None): 

    data_dir = dexhub.download_dataset(uuid=dataset_uuid, load_dir = "./", load_to_mem = False)
    dex_data_dir = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dex')]

    mujoco_model = get_sim(dex_data_dir)
    mujoco_data = mujoco.MjData(mujoco_model)

    renderer = mujoco.Renderer(mujoco_model) 

    data = dexhub.load(dex_data_dir[0])
    obs_dim = data.data[0].obs.mj_qpos.shape
    act_dim = data.data[0].act.mj_ctrl.shape

    robot_cfg = init_hydra_config("./panda.yaml")
    robot = make_robot(robot_cfg)

    # create random name for the dataset  (if not provided)
    # style:  xxxx/xxxx 
    if dataset_name is None:
        dataset_name = "younghyopark/" + str(np.random.randint(0, 1000))

    try: 
        dataset = LeRobotDataset.create(dataset_name, fps = 30, robot = robot)
    except: 
        # add random index to the dataset_name
        dataset_name = dataset_name + str(np.random.randint(0, 1000))
        dataset = LeRobotDataset.create(dataset_name, fps = 30, robot = robot)

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
                'observation.images.main': pixels, 
                'action': act
            })
        # Now try saving
        dataset.save_episode(task="test", encode_videos=False)

    dataset.consolidate()
    
    return mujoco_model, dataset
