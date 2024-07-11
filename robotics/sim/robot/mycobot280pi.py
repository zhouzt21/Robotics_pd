import os
import sapien
import numpy as np
from .urdf.urdf_hub import mycobot_pi, mycobot_pi_v2
from .robot_base import Robot, MobileRobot
from .controller import PDJointVelController, MimicPDJointPosController, PDJointVelConfig, PDJointPosConfig
from .mobile_controller import PDBaseVelController
from .urdf import URDFTool
from ..simulator import Simulator
from robotics.sim import get_engine
from sapien import Scene

from robotics import Pose
from sapien.physx import PhysxArticulation, PhysxRigidStaticComponent


from .posvel_controller import PDJointPosVelController, PDJointPosVelConfig


def add_loop_for_gripper(robot: PhysxArticulation, scene: Scene, pmodel=None):
    if pmodel is None:
        pmodel = robot.create_pinocchio_model() # type: ignore
    pmodel.compute_forward_kinematics(np.zeros(robot.dof))

    # robot.set_qpos(np.zeros(robot.dof)) # type: ignore
    def get_pose(link: sapien.physx.PhysxArticulationLinkComponent):
        return pmodel.get_link_pose(link.index)

    flange = robot.find_link_by_name("joint6_flange")

    lik = robot.find_link_by_name("gripper_left2")
    lok = robot.find_link_by_name("gripper_left3")
    lf = robot.find_link_by_name("gripper_left1")
    T_pw = get_pose(lf).inv().to_transformation_matrix()
    p_w = get_pose(lf).p + (get_pose(lik).p - get_pose(lok).p) # type: ignore

    # builder = scene.create_actor_builder()
    # builder.add_sphere_visual(radius=0.01, material=(1, 0, 0))
    # actor = builder.build_kinematic(name="vis")
    # actor.set_pose(Pose(p_w))

    T_fw = get_pose(lik).inv().to_transformation_matrix()
    p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
    p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
    drive = scene.create_drive(lik, Pose(p_f), lf, Pose(p_p))
    drive.set_limit_x(0, 0)
    drive.set_limit_y(0, 0)
    drive.set_limit_z(0, 0)

    rik = robot.find_link_by_name("gripper_right2")
    rok = robot.find_link_by_name("gripper_right3")
    rf = robot.find_link_by_name("gripper_right1")
    T_pw = get_pose(rf).inv().to_transformation_matrix()
    p_w = get_pose(rf).p + (get_pose(rik).p - get_pose(rok).p) # type: ignore
    T_fw = get_pose(rik).inv().to_transformation_matrix()
    p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
    p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
    drive = scene.create_drive(rik, Pose(p_f), rf, Pose(p_p))
    drive.set_limit_x(0, 0)
    drive.set_limit_y(0, 0)
    drive.set_limit_z(0, 0)

    # gear = scene.create_gear(lok, Pose(), rok, Pose(q=[0, 0, 0, 1]))
    # gear.gear_ratio = -1
    # gear.enable_hinges()

    for idx, l in enumerate([lik, lok, lf, rik, rok, rf, flange]):
        if l is not None:
            for s in l.collision_shapes:
                s.set_collision_groups([1, 1, 4, 0])
    return [j.name for j in [lik, lok, lf, rik, rok, rf, flange] if j is not None]



class MyCobot280Arm(MobileRobot):
    ignore_srdf = True

    def __init__(self, control_freq: int, 
                 balance_passive_force: bool = True, 
                 fix_root_link=True, 
                 arm_controller='posvel', 
                 add_base=True, move_base=True,
                 add_camera=True,
                 ) -> None:
        self.add_camera = add_camera
        urdf_path, urdf_tool = mycobot_pi_v2(add_base, move_forward=move_base)
        self.add_base = add_base
        self.move_base = move_base

        super().__init__(control_freq, urdf_path, {}, balance_passive_force, fix_root_link)
        joints = urdf_tool.all_joints.values()

        active_joints = []
        for i in joints:
            if i.elem.get('type') != 'fixed':
                active_joints.append(i.name)
            

        self.arm_joint_names = [f'joint{i+2}_to_joint{i+1}' for i in range(5)]
        self.arm_joint_names += ['joint6output_to_joint6']

        self.gripper_joint_names = ['gripper_controller', 'gripper_base_to_gripper_right3']


        base_joint_names = []
        for i in ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint']:
            if i in urdf_tool.all_joints: 
                if urdf_tool.all_joints[i].elem.attrib['type'] != 'fixed':
                    base_joint_names.append(i)
        self.base_joint_names = base_joint_names


        if arm_controller == 'vel':
            controller = PDJointVelController(
                PDJointVelConfig(
                    lower=[-5.] * 6,
                    upper=[5.] * 6,
                    stiffness=0,#1000,
                    damping=1000.,
                    force_limit=100.,
                    friction=10,
                ), 
                self, self.arm_joint_names
            )
        elif arm_controller == 'posvel':
            controller = PDJointPosVelController(
                PDJointPosVelConfig(
                    lower=[-3.14] * 6,
                    upper=[3.14] * 6,
                    stiffness=0,
                    damping=10000., force_limit=100., friction=10,
                    Kp=0.1
                ), 
                self, self.arm_joint_names
            )
        elif arm_controller.startswith('ee_delta'):
            from .ee_delta_controller import EEDeltaPoseController, EEDeltaPoseConfig
            controller = EEDeltaPoseController(
                EEDeltaPoseConfig(
                    stiffness=500., damping=0., force_limit=100., friction=10, use_target=arm_controller == 'ee_delta_target'
                ),
                self, self.arm_joint_names, 'gripper_base', 'base_link'
            )
        else:
            raise NotImplementedError(f"Unknown controller {arm_controller}")

        if self.move_base:
            self.register_controller(
                "base", PDBaseVelController(PDJointPosConfig(
                    lower=[-1., -1., -3.14],
                    upper=[1., 1., 3.14],
                    stiffness=10.,
                    damping=5000,
                    force_limit=2000,
                ), self, self.base_joint_names)
            )

        self.register_controller('arm', controller)
        self.register_controller(
            "gripper", MimicPDJointPosController(PDJointPosConfig(
                lower=-0.7,  # a trick to have force when the object is thin
                upper=0.15,
                stiffness=1000,
                damping=5,
                force_limit=10.,

            ), self, self.gripper_joint_names)
        )


    def get_sensors(self):
        if self.add_camera:
            self.lidar = self.build_lidar(base='robot/rplidar_link')

            from ..sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig
            self.depth = StereoDepthCamera(StereoDepthCameraConfig(p=(0.2, 0, 0.05), base=f'robot/base_link'))
            return {
                'lidar': self.lidar,
                'depth': self.depth,
            }
        else:
            return {}


    def get_ros_plugins(self):
        from ..ros_plugins.ros_plugin import ArmServo, CMDVel, TFPublisher, LidarPublisher, RobotNamePublisher, RGBDPublisher
        assert isinstance(self.controllers['arm'], PDJointPosVelController)

        if self.move_base:
            frame_mapping= {
                'base_link': 'robot/' + 'base_link',
                'odom': 'world',
                'base_laser': 'robot/rplidar_link',
                'realsense': 'depth'
            }
            plugins =  [
                RobotNamePublisher('mycobot280pi'),
                CMDVel(action_decay=0.1),
                LidarPublisher(self.lidar, 'base_laser', frame_mapping=frame_mapping),
                TFPublisher(
                    dynamic=[("odom", 'base_link')],
                    static=[('base_link', 'base_laser'), ('base_link', 'realsense')],
                    frame_mapping=frame_mapping
                ),
                RGBDPublisher(self.depth, frame_name='realsense', frame_mapping=frame_mapping, frequence=10) #TODO: fix the frequence
            ]
        else:
            plugins = []
        plugins.append(
            ArmServo(self.controllers['arm'])
        )
        return plugins


    def _update_loader(self, loader):
        right = 'gripper_right1'
        left = 'gripper_left1'
        loader.set_material(0.3, 0.3, 0.0)
        loader.set_link_material(left, 1.0, 1.0, 0.0)
        loader.set_link_material(right, 1.0, 1.0, 0.0)
        loader.set_link_patch_radius(left, 0.05)
        loader.set_link_patch_radius(right, 0.05)
        loader.set_link_min_patch_radius(left, 0.05)
        loader.set_link_min_patch_radius(right, 0.05)


    def _post_load(self, world: Simulator):
        import warnings
        scene = world._scene

        if not scene.physx_system.config.enable_tgs:
            warnings.warn(
                "TGS is not enabled in scene. TGS is recommended for simulating loop joints."
            )
        if scene.physx_system.config.solver_iterations < 15:
            warnings.warn(
                f"Solver iteration ({scene.physx_system.config.solver_iterations}) of this sceen is probably too small for simulating XArm"
            )


        self.robot = self.articulation
        
        arm_idx, self.arm_joints = self.get_active_joints(self.arm_joint_names)
        self.left_gripper_joint = self.robot.find_link_by_name(
            "gripper_left3"
        ).joint
        self.right_gripper_joint = self.robot.find_link_by_name(
            "gripper_right3"
        ).joint

        assert self.left_gripper_joint.name == "gripper_controller"



        if not self.add_base:
            ignore_set = [
                ('g_base', 'joint1', 'joint2'),
            ]
        else:
            ignore_set = [
                ('base_link', 'joint1', 'right_wheel', 'left_wheel', 'rplidar_link', 'front_caster_link', 'ground')
            ]
        name2link = {i.name: i for i in self.articulation.get_links()}

        if hasattr(world, 'ground'):
            for i in world.ground.components:
                if isinstance(i, PhysxRigidStaticComponent):
                    name2link['ground'] = i # type: ignore

        group_id = 4
        for collision_ignorance in ignore_set:
            for k in collision_ignorance:
                #assert k in name2link, f"{k} not in {name2link.keys()}"
                if k not in name2link:
                    import logging
                    logging.warning(f"{k} not in {name2link.keys()}")
                    continue
                link = name2link[k]
                _cs = link.get_collision_shapes()
                assert len(_cs) == 1, f"{k} has no collision shape"
                cs = _cs[0]
                cg = cs.get_collision_groups()
                cg[2] = (1<<group_id)
                cs.set_collision_groups(cg)
            group_id += 1

        # # fingers = self._create_drives(world._scene)
        # if not hasattr(self, 'pmodel'):
        #     self.pmodel = self.robot.create_pinocchio_model() # type: ignore
        fingers = add_loop_for_gripper(self.robot, scene, self.pmodel)

        engine = get_engine()

        #TODO: set arm and gripper pd
        #qpos = engine.get_qpos(self.robot)
        qpos = np.zeros(self.robot.dof)
        qpos0 = [-1.302190154912969, -0.7438593271999832, -1.1550588989698474, 0.3680899392456041, -0.061261056745000965, -0.3389429407372988]

        qpos[arm_idx[0]:] = qpos0 + [0] * 6
        #self.robot.set_qpos(qpos)
        engine.set_qpos(self.robot, qpos)
        if self.add_base:
            engine.set_root_pose(self.robot, Pose([0, 0, 0.0]))
        else:
            engine.set_root_pose(self.robot, Pose([0, 0, 0.1]))

        
        if world._scene_idx == 0:
            srdf_path = self.get_srdf_path()
            content = self.generate_srdf([ignore_set[0][:-2]] + [fingers], 'mycobot_urdf_')
            with open(srdf_path, 'w') as f:
                f.write(content)


    @property
    def base_name(self):
        return 'base_link'

    @property
    def ee_name(self):
        return 'gripper_base'