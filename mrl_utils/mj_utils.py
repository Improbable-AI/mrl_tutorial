import dexhub
import mujoco 
import mediapy as media

def get_sim(data_list):
    data = data_list[0]
    data = dexhub.load(data)
    model = data.get_mjmodel()
    return model 

def view_trajectory(sim, episode_traj): 

    model = sim 
    data = mujoco.MjData(model)

    frames = [] 
    mujoco.mj_resetData(model, data)
    # Make renderer, render and show the pixels
    with mujoco.Renderer(model) as renderer:
        for i in range(len(data.data)):
            obs = data.data[i].obs.mj_qpos
            act = data.data[i].act.mj_ctrl
            sim.step(act)
            sim.render()
    return sim