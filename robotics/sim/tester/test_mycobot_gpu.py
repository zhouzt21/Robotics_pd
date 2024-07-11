import os

import time
import numpy as np
from robotics.sim import Simulator, SimulatorConfig, CameraConfig, GPUEngineConfig, CameraV2
from robotics.sim.robot.mycobot280pi import MyCobot280Arm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--not_view', action='store_true')
args = parser.parse_args()

camera = CameraV2(CameraConfig(look_at=(0., 0., 0.5), p=(1.5, 1.5, 2.)))
sim = Simulator(
    SimulatorConfig(
        viewer_camera=CameraConfig(look_at=(0., 0., 0.5), p=(0.5, 0.5, 1.)),
        gpu_config=GPUEngineConfig()
    ),
    MyCobot280Arm(60, arm_controller='posvel', add_camera=False), {'camera': camera}, # ros_node=node
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
    else:
        import cv2
        img = sim._engine.take_picture(0)['camera']['Color']
        cv2.imshow('img', img)
        cv2.waitKey(1)
    time.sleep(max(dt - (time.time() - cur), 0.))