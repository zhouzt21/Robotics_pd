import os
os.environ["ROS_DOMAIN_ID"] = "1"

import time
import numpy as np
from robotics.sim import Simulator, SimulatorConfig, CameraConfig
from robotics.sim.robot.mycobot280pi import MyCobot280Arm

from robotics.ros import ROSNode
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--not_view', action='store_true')
args = parser.parse_args()

# node = ROSNode('cobot_sim', use_sim_time=False)

sim = Simulator(
    SimulatorConfig(viewer_camera=CameraConfig(look_at=(0., 0., 0.5), p=(0.5, 0.5, 1.))),
    MyCobot280Arm(60, arm_controller='posvel'), {}, # ros_node=node
)

sim.reset()
dt = sim.dt


idx = 0
while (args.not_view) or (not sim.viewer.closed):
    cur = time.time()
    idx += 1
    action = np.zeros((7,))
    action[-1] = (idx // 100)%2 * 2 - 1
    sim.step(None)
    if not args.not_view:
        sim.render()
    time.sleep(max(dt - (time.time() - cur), 0.))