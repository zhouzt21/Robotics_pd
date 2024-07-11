import sapien
import numpy as np
from sapien import physx
from robotics.utils.path import PACKAGE_ASSET_DIR
from robotics.sim import Simulator
from .robot_base import MobileRobot
from .controller import MimicPDJointPosController, PDJointPosConfig, PDJointPosController
from .mobile_controller import PDBaseVelController
from robotics.utils.sftp import download_if_not_exists


class MobileSapien(MobileRobot):
    def __init__(self, control_freq: int, balance_passive_force: bool = True, motion_model='holonomic') -> None:
        assets_path = [
            (f"{PACKAGE_ASSET_DIR}/mobile_sapien/mobile_manipulator_description", "mobile_sapien/mobile_manipulator_description"),
            (f"{PACKAGE_ASSET_DIR}/mobile_sapien/xarm_description", "mobile_sapien/xarm_description"),
            (f"{PACKAGE_ASSET_DIR}/mobile_sapien/xarm7/xarm_urdf", "xarm7/sapien_xarm7/xarm_urdf"),
        ]
        for k, v in assets_path:
            download_if_not_exists(k, v)

        urdf_path = f"{PACKAGE_ASSET_DIR}/mobile_sapien/mobile_manipulator_xy.urdf"
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
        self.arm_joint_names = [f'joint{i+1}' for i in range(7)]
        self.gripper_joint_names = ['left_outer_knuckle_joint', 'right_outer_knuckle_joint']


        super().__init__(control_freq, urdf_path, urdf_config, balance_passive_force, fix_root_link=True, motion_model=motion_model)


        self.register_controller(
            "base", PDBaseVelController(PDJointPosConfig(
                lower=[-1., -1., -3.14],
                upper=[1., 1., 3.14],
                stiffness=10,
                damping=50000,
                force_limit=20,
                mode='acceleration'
            ), self, self.base_joint_names)
        )

        self.register_controller(
            'arm', PDJointPosController(PDJointPosConfig(
                lower=-3.14,
                upper=3.14,
                stiffness=1000,
                # damping=20,
                # force_limit=38,
                damping=100,
                force_limit=10,
                mode='acceleration'
            ), self, self.arm_joint_names)
        )

        self.register_controller(
            "gripper", MimicPDJointPosController(PDJointPosConfig(
                lower=-0.01,  # a trick to have force when the object is thin
                upper=0.86,
                stiffness=8000,
                damping=1e2,
                force_limit=10.,
                mode='acceleration'

            ), self, self.gripper_joint_names)
        )

        
    def _update_loader(self, loader):

        loader.set_material(0.3, 0.3, 0.0)
        loader.set_link_material("left_finger", 1.0, 1.0, 0.0)
        loader.set_link_material("right_finger", 1.0, 1.0, 0.0)
        loader.set_link_patch_radius("left_finger", 0.05)
        loader.set_link_patch_radius("right_finger", 0.05)
        loader.set_link_min_patch_radius("left_finger", 0.05)
        loader.set_link_min_patch_radius("right_finger", 0.05)


    def _post_load(self, world: Simulator):
        self.robot = self.articulation

        
        arm_idx, self.arm_joints = self.get_active_joints(self.arm_joint_names)
        gripper_idx, _ = self.get_active_joints(self.gripper_joint_names)
        #self.robot.active_joints[:7]
        self.left_gripper_joint = self.robot.find_link_by_name(
            "left_outer_knuckle"
        ).joint
        self.right_gripper_joint = self.robot.find_link_by_name(
            "right_outer_knuckle"
        ).joint

        assert self.left_gripper_joint.name == "left_outer_knuckle_joint"


        ignore_set = [
            ('gimbal_base', 'gimbal_pitch', 'realsense'),
            ('base_link', 'link1', 'link2'),
        ]
        name2link = {i.name: i for i in self.articulation.get_links()}

        group_id = 1
        for collision_ignorance in ignore_set:
            for k in collision_ignorance:
                link = name2link[k]
                _cs = link.get_collision_shapes()
                assert len(_cs) == 1, f"{k} has no collision shape"
                cs = _cs[0]
                cg = cs.get_collision_groups()
                cg[2] = (1<<group_id)
                cs.set_collision_groups(cg)
            group_id += 1

        self._create_drives(world._scene)

        qpos = self.robot.get_qpos()
        qpos[arm_idx] = [0, 0, 0, np.pi / 3, 0, np.pi / 3, -np.pi / 2]

        qpos[gripper_idx] = [0.5] * 2
        self.robot.set_qpos(qpos)

        self.robot.set_root_pose(sapien.Pose(p=[0, 0, 0.1]))

    def get_sensors(self):
        #TODO: check the correctness of the lidar frame and add the depth camera
        self.lidar = self.build_lidar(base='robot/velodyne')
        from ..sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig
        self.depth = StereoDepthCamera(StereoDepthCameraConfig(p=(0., 0, 0.), base=f'robot/realsense'))
        return {
            'lidar': self.lidar,
            'depth': self.depth,
        }

    @property
    def frame_mapping(self):
        return {
            'base_link': 'robot/' + self.base_name,
            'odom': 'world',
            'base_laser': 'robot/velodyne',
            'realsense': 'robot/realsense',
        }


    def get_ros_plugins(self):
        from ..ros_plugins import CMDVel, TFPublisher, LidarPublisher, RGBDPublisher
        frame_mapping = self.frame_mapping
        return [
            CMDVel(action_decay=0.9), # NOTE: action decay some how imitates the inertia of the robot and would affect the speed of the robot
            TFPublisher(
                dynamic=[("odom", 'base_link')],
                static=[('base_link', 'base_laser'), ('base_link', 'realsense')],
                frame_mapping=frame_mapping
            ),
            LidarPublisher(
                self.lidar, 'base_laser', frame_mapping=frame_mapping
            ),
            RGBDPublisher(self.depth, frame_name='realsense', frame_mapping=frame_mapping)
        ]
        

    @property
    def ee_name(self):
        return 'link_tcp'

    @property
    def base_name(self):
        return 'base_link'

        
        
    def _create_drives(self, scene):
        self.robot.set_qpos(np.zeros(self.robot.dof)) # type: ignore

        lik = self.robot.find_link_by_name("left_inner_knuckle")
        lok = self.robot.find_link_by_name("left_outer_knuckle")
        lf = self.robot.find_link_by_name("left_finger")
        T_pw = lf.pose.inv().to_transformation_matrix()
        p_w = lf.pose.p + lik.pose.p - lok.pose.p # type: ignore
        T_fw = lik.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
        drive = scene.create_drive(lik, sapien.Pose(p_f), lf, sapien.Pose(p_p))
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)

        rik = self.robot.find_link_by_name("right_inner_knuckle")
        rok = self.robot.find_link_by_name("right_outer_knuckle")
        rf = self.robot.find_link_by_name("right_finger")
        T_pw = rf.pose.inv().to_transformation_matrix()
        p_w = rf.pose.p + rik.pose.p - rok.pose.p # type: ignore
        T_fw = rik.pose.inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
        drive = scene.create_drive(rik, sapien.Pose(p_f), rf, sapien.Pose(p_p))
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)

        # NOTE: the gear is very unstable
        # gear = scene.create_gear(lok, sapien.Pose(), rok, Pose(q=[0, 0, 0, 1]))
        # gear.gear_ratio = -1
        # gear.enable_hinges()

        for l in [lik, lok, lf, rik, rok, rf]:
            for s in l.collision_shapes:
                s.set_collision_groups([1, 1, 4, 0])