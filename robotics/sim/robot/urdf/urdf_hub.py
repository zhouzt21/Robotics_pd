"""_summary_
generating the path to the urdf file
used to register robot
"""

import numpy as np
from xml.etree.ElementTree import tostring 
import os
from robotics.utils.sftp import download_if_not_exists
from robotics.utils.path import PACKAGE_ASSET_DIR
from .urdf import URDFTool


def mycobot_pi():
    path = os.path.join(PACKAGE_ASSET_DIR, "mycobot_pi")
    download_if_not_exists(path, "mycobot_ros2/mycobot_description/urdf/mycobot_pi")
    urdf_path = os.path.join(path, "mycobot_urdf.urdf")
    urdf_tool = URDFTool.from_path(urdf_path)
    for i in urdf_tool.root.iter():
        if 'filename' in i.attrib:
            i.attrib['filename'] = i.attrib['filename'].replace('mycobot_description/urdf/mycobot_pi/', '')
        
    new_path = os.path.join(urdf_tool.package_dir, 'mycobot_urdf_.urdf')
    urdf_tool.export(new_path)
    print('write to', new_path)

    return new_path, urdf_tool

    
def turtlebot4_xy():
    path = os.path.join(PACKAGE_ASSET_DIR, "turtlebot4")
    download_if_not_exists(path, "turtlebot4")
    urdf_path = f"{PACKAGE_ASSET_DIR}/turtlebot4_xy.urdf"
    turtlebot = URDFTool.from_path(urdf_path)
    return urdf_path, turtlebot


def mycobot_pi_v2(add_base=True, move_forward=False):
    path = os.path.join(PACKAGE_ASSET_DIR, "mycobot_pi_v2")
    download_if_not_exists(path, "mycobot_ros/mycobot_description/urdf/mycobot")
    urdf_path = os.path.join(path, "mycobot_with_gripper_parallel.urdf")
    urdf_tool = URDFTool.from_path(urdf_path)
    for i in urdf_tool.root.iter():
        if 'filename' in i.attrib:
            i.attrib['filename'] = i.attrib['filename'].replace('mycobot_description/urdf/mycobot/', '')

    for k, v in urdf_tool.all_joints.items():
        if k.startswith('gripper'):
            #print(k, v.elem.find('axis').attrib['xyz'])
            limit = v.elem.find('limit')
            assert limit is not None

            lower, upper = float(limit.attrib['lower']), float(limit.attrib['upper'])
            if 'right' in k:
                v.elem.find('axis').attrib['xyz'] = '0 0 -1' # type: ignore
                upper, lower = -lower, -upper

            if k == 'gripper_right3_to_gripper_right1' or k == 'gripper_left3_to_gripper_left1':
                lower, upper = str(lower), str(upper)
                lower = lower.replace('0.5', '0.7')
                upper = upper.replace('0.5', '0.7')

            limit.attrib['lower'] = str(lower)
            limit.attrib['upper'] = str(upper)

        
    package_dir = urdf_tool.package_dir
    if add_base:
        urdf_tool.remove('g_base_to_joint1')
        urdf_tool = urdf_tool.prune_from('joint1')

        turtlebot = turtlebot4_xy()[1]

        if not move_forward:
            turtlebot.remove('root_z_rotation_joint')
            turtlebot = turtlebot.prune_from('base_link')
        # else:
        #     for k, v in turtlebot.all_joints.items():
        #         if k == 'root_y_axis_joint':
        #             v.elem.attrib['type'] = 'fixed'
        #         if k == 'root_z_rotation_joint':
        #             v.elem.attrib['type'] = 'fixed'

        turtlebot.remove('rplidar_joint')
        base_link = turtlebot.prune_from('base_link')
        lidar = turtlebot.prune_from('rplidar_link')

        base_link.remove('shell_link_joint')
        base_link = base_link.prune_from('base_link')
        from robotics import Pose
        import transforms3d
        quat = transforms3d.euler.euler2quat(0, 0, np.pi)

        base_link =  base_link.add(lidar, Pose((0.0, 0.1, 0.098715 + 0.04), quat), 'base_link', 'rplidar')


        urdf_tool.remove('joint6output_to_gripper_base')
        arm = urdf_tool.prune_from('joint1')
        gripper = urdf_tool.prune_from('gripper_base')
        quat = transforms3d.euler.euler2quat(1.579, 0., np.pi/4)
        urdf_tool = arm.add(gripper, Pose((0, 0, 0.034), quat), 'joint6_flange', 'gripper_base')

        quat = transforms3d.euler.euler2quat(0, 0, 1.5707963267948966)
        urdf_tool = base_link.add(urdf_tool, Pose((0., 0., 0.095), quat), 'base_link', 'joint1')

        for i in urdf_tool.all_links.values():
            if i.name == 'gripper_left1':
                i.elem.find('collision').find('origin').attrib['xyz'] = i.elem.find('visual').find('origin').attrib['xyz'] # type: ignore

        # NOTE: remove the realsense camera link ..
        urdf_tool.remove('realsense_joint')
        urdf_tool = urdf_tool.prune_from('joint1')


    new_path = os.path.join(package_dir, 'mycobot_urdf_.urdf')
    urdf_tool.export(new_path)
    print('write to', new_path)

    return new_path, urdf_tool


if __name__ == "__main__":
    mycobot_pi()