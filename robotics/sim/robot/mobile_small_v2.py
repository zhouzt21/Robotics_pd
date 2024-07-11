"""
Mobile small v2
"""
import numpy as np
from robotics.utils.path import PACKAGE_ASSET_DIR
from .robot_base import Robot, MobileRobot
from .controller import PDJointVelController, MimicPDJointPosController, PDJointVelConfig, PDJointPosConfig
from .mobile_controller import PDBaseVelController
from robotics import Pose
from typing import Any


class MobileSmallV2(MobileRobot):
    def __init__(self, control_freq: int, balance_passive_force: bool = True, motion_model='holonomic') -> None:

        urdf_path = f"{PACKAGE_ASSET_DIR}/mobile_small.urdf"
        urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                right_panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                right_panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )

        self.base_joint_names = ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"]
        self.arm_joint_names = [f'right_panda_joint{i+1}' for i in range(7)]
        self.gripper_joint_names = [f"right_panda_finger_joint{i+1}" for i in range(2)]

        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        self.arm_joint_delta = 0.1
        self.arm_ee_delta = 0.1

        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100


        super().__init__(control_freq, urdf_path, urdf_config, balance_passive_force, fix_root_link=True, motion_model=motion_model)


        self.register_controller(
            "base", PDBaseVelController(PDJointPosConfig(
                lower=[-1., -1., -3.14],
                upper=[1., 1., 3.14],
                damping=1000,
                force_limit=2000,
            ), self, self.base_joint_names)
        )

        self.register_controller(
            'arm', PDJointVelController(PDJointVelConfig(
                lower=-3.14,
                upper=3.14,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
            ), self, self.arm_joint_names)
        )

        self.register_controller(
            "gripper", MimicPDJointPosController(PDJointPosConfig(
                lower=-0.01,  # a trick to have force when the object is thin
                upper=0.04,
                stiffness=self.gripper_stiffness,
                damping=self.gripper_damping,
                force_limit=self.gripper_force_limit,
            ), self, self.gripper_joint_names)
        )



    def get_ros_plugins(self):
        from ..ros_plugins import CMDVel, TFPublisher, LidarPublisher
        frame_mapping={
            'base_link': 'robot/' + self.base_name,
            'odom': 'world',
            'base_laser': 'robot_base',
        }
        return [
            CMDVel(),
            TFPublisher(
                dynamic=[("odom", 'base_link')],
                static=[('base_link', 'base_laser')],
                frame_mapping=frame_mapping
            ),
            LidarPublisher(
                self.lidar, 'base_laser', frame_mapping=frame_mapping
            )
        ]
        
        


    def get_state(self):
        state = self._get_state()
        state['robot_root_pose'] = np.concatenate([state['robot_root_pose'].p, state['robot_root_pose'].q])
        return state

    @property
    def ee_name(self):
        return 'right_panda_hand'

    @property
    def base_name(self):
        return 'mobile_base'