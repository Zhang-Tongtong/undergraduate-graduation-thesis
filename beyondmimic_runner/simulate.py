import pickle
import mujoco
import numpy as np
import mujoco.viewer


def run():

    data=pickle.load(open("trajectory.pkl","rb"))

    qpos=data["qpos"]

    model=mujoco.MjModel.from_xml_path(
        "BeyondMimic/data/humanoid.xml"
    )

    data_sim=mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model,data_sim) as viewer:

        for i in range(len(qpos)):

            data_sim.qpos[:len(qpos[i])] = qpos[i]

            mujoco.mj_step(model,data_sim)

            viewer.sync()


if __name__=="__main__":
    run()