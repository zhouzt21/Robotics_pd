import os
from sapien import Pose
os.environ["ROS_DOMAIN_ID"] = "1"

import time
import numpy as np
from robotics.sim import Simulator, SimulatorConfig, CameraConfig
from robotics.sim.robot.xarm7 import XArm7

from robotics.ros import ROSNode
import argparse

robot1 = XArm7(60)
robot2 = XArm7(60)

sim = Simulator(
    SimulatorConfig(viewer_camera=CameraConfig(look_at=(0., 0., 0.5), p=(0.5, 0.5, 1.))),
    robot1, {'robot2': robot2}, # ros_node=node
)

sim.reset()
dt = sim.dt
robot2.articulation.set_pose(Pose([0., 2., 0.]))


idx = 0
while (not sim.viewer.closed):
    cur = time.time()
    idx += 1
    action = np.zeros((8,))
    #action[0] = 0.5
    action[:7] = np.array([0, 0, 0, 1. / 3, 0, 1. / 3, -1. / 2])
    # action[-2] = 1.
    action[-1] = (idx // 100)%2 * 2 - 1
    robot2.set_action(action)
    sim.step(action)
    sim.render()
    time.sleep(max(dt - (time.time() - cur), 0.))