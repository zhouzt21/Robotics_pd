from typing import Dict, cast
import subprocess
import numpy as np
import torch
from robotics.sim import Simulator, Camera, MobileSmallRobot
from typing import Dict, cast
from robotics.utils import tile_images
import sapien.core as sapien
from robotics.utils import logger

from . import nav_sensor_publisher
from .nav_sensor_publisher import NavSensorPublisher, NavSensorPublisherConfig, pose2tf
from .rgb_publisher import RGBPublisher, RGBPublisherConfig
from robotics.planner.skills import KeyboardController, show_camera
from robotics.planner import MobileAgent


def run_with_ros(
    sim: Simulator, 
    wait_for_goal: bool = False,
    record: bool = False,
):
    if record:
        command = ['ros2', 'bag', 'record', '-a']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    from .time_publisher import TimePublisher
    clock = TimePublisher(sim.dt)

    robot = cast(MobileSmallRobot, sim.robot)

    camera_name = 'robot_base'
    agent = MobileAgent()
    agent.load(sim, robot)

    action = sim.action_space.sample() * 0
    action[:2] = [1., 0.]


    nav_publisher = NavSensorPublisher(NavSensorPublisherConfig(channel='/tf,/odom,/scan,/cloud'))
    rgb_publisher = RGBPublisher(RGBPublisherConfig())

    camera_names = ['robot_base', 'robot_base_2', 'robot_base_3', 'robot_base_4']
    cameras: Dict[str, Camera] = {name: cast(Camera, sim.elements[name]) for name in camera_names}

    task = KeyboardController()

    import tqdm

    trange = tqdm.trange(1000)

    show =  show_camera(cameras=['robot_base'], topic='camera', use_cv2=True)

    with logger.configure(2, dump=False) as L:
        if not wait_for_goal:
            todo = [task, show]
        else:
            from robotics.ros.playground.cmd_vel_controller import CmdVelSkill, CmdVelSkillConfig
            todo = [CmdVelSkill(CmdVelSkillConfig()), show]

        try:
            while True:
                trange.update(1)
                action = agent.act({}, *todo)
                todo = []
                sim.step(action)
                clock.step()
                sim.render()


                observations = sim.elements.get_observation()
                camera_name = 'robot_base'
                camera = cameras[camera_name]
                tf = camera.get_pose()
                # base_tf = robot.get_base_pose()

                points = []

                # params = camera.get_camera_params()
                # import numpy as np
                # extrinsic_cv = np.r_[params['extrinsic_cv'], [[0, 0, 0, 1]]]
                # instrinsic_cv = params['intrinsic_cv']

                # pose = camera.get_pose().to_transformation_matrix()
                # cam2world_gl = params['cam2world_gl']
                # print(np.linalg.inv(pose) @ cam2world_gl)
                # print(pose)
                # params['cam2world_gl'] = pose @ extrinsic_cv


                camera_params = {name: camera.get_camera_params() for name, camera in cameras.items()}
                for name in camera_names:
                    points.append(nav_sensor_publisher.depth2pts(observations[name], camera_params[name]))
                points = torch.cat(points, dim=0) # changed in maniskill3

                laser = nav_sensor_publisher.pointcloud2laser(points, tf, verbose=True)
                pointcloud = nav_sensor_publisher.depth2pointcloud_msg(observations[camera_name], camera_params[camera_name], tf)

                vel = agent.articulated.get_qvel()
                camera_local = sapien.Pose()

                nav_publisher.publish((laser, pointcloud), (tf, camera_local), vel, not wait_for_goal) # if not args.wait_for_goal then add odom

                # pos = observations[camera_name]['Position'].copy()
                # pos = pos / np.clip(pos[:, :, 2:3], -np.inf, -1e-10)
                # image= pos[..., :3] @ camera_params[camera_name]['intrinsic_cv'].T
                # raise KeyboardInterrupt


                rgb = np.uint8(observations[camera_name]['Color'][..., :3] * 255)
                depth = observations[camera_name]['Position'][:, :, 2]
                extrinsic = np.linalg.inv(tf.to_transformation_matrix())  @ camera_params[camera_name]['cam2world_gl']
                intrinsic = camera_params[camera_name]['intrinsic_cv']
                rgb_publisher.publish(rgb, depth, extrinsic, intrinsic)

                if task._terminated:
                    break

        except KeyboardInterrupt:
            # L.animate('camera', 'test.mp4', fps=30)
            del nav_publisher
            del rgb_publisher
            del clock