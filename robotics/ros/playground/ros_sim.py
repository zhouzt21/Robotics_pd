import numpy as np
import torch
import time
import open3d as o3d
import cv2
from robotics.sim import Simulator, SimulatorConfig, Camera, CameraConfig, PoseConfig, MobileSmallRobot
from robotics.sim.environs.ycb_clutter import YCBClutter, YCBConfig
from typing import Dict, List, Tuple, cast, Union
from robotics.utils import tile_images
import sapien.core as sapien


from robotics.sim.robot.mobile_small_v2 import MobileSmallV2
from robotics.utils import logger


import argparse
from robotics.ros.playground.run_sim import run_with_ros


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='thor')
    parser.add_argument('--wait_for_goal', action='store_true', default=False)
    parser.add_argument('--no_ycb', action='store_true', default=False)
    parser.add_argument('--record', action='store_true', default=False)
    args = parser.parse_args()


    # robot = MobileSmallRobot(
    #     RobotConfig(
    #         control_freq=20,
    #         control_mode='base_pd_joint_vel_arm_pd_joint_vel',
    #         fix_root_link=True
    #     )
    # )
    robot = MobileSmallV2(control_freq=20)

    if args.env == 'wall':
        from robotics.sim.environs import SapienWallEnv, WallConfig
        environ = SapienWallEnv(config=WallConfig(maze_id=2))
    elif args.env == 'thor':
        from robotics.sim.environs.thor import ThorEnv, ThorEnvConfig
        environ = ThorEnv(config=ThorEnvConfig())
    else:
        raise NotImplementedError


    camera_cfg = CameraConfig(pose=PoseConfig((0.0, -0., 1.8), (0.707, 0., 0.707, 0.)), actor_uid="robot/mobile_base")
    elements = {}
    elements['environ'] = environ
    if not args.no_ycb:
        ycb = YCBClutter(config=YCBConfig(N=5))
        elements['ycb'] = ycb
    elements['look_down'] = Camera(camera_cfg)

    sim = Simulator(
        SimulatorConfig(viewer_camera=CameraConfig(pose=PoseConfig((0.8, -0.8, 1.8), (0.707, 0., 0.707, 0.))),), 
        robot,  elements, add_ground=(args.env == 'wall')
    )

    sim.reset()
    sim.update_render()
    robot.set_base_pose((0.8, -0.8), 0.5)

    run_with_ros(sim, wait_for_goal=args.wait_for_goal, record=args.record)