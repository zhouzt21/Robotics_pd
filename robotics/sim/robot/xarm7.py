import numpy as np
from .robot_base import Robot
from .controller import MimicPDJointPosController, PDJointPosConfig, PDJointPosController
from .urdf import URDFTool
from robotics.utils.path import PACKAGE_ASSET_DIR
from robotics.sim import Simulator
from robotics import Pose

def xarm7_gripper():
    urdf_path = f"{PACKAGE_ASSET_DIR}/sapien_xarm7/xarm7_d435.urdf"
    xarm7 = URDFTool.from_path(urdf_path)
    return urdf_path, xarm7


class XArm7(Robot):
    def __init__(self, control_freq: int, balance_passive_force: bool = True, fix_root_link=True) -> None:
        urdf_path, urdf_tool = xarm7_gripper()

        super().__init__(control_freq, urdf_path, {}, balance_passive_force, fix_root_link)
        joints = urdf_tool.all_joints.values()

        active_joints = []
        for i in joints:
            if i.elem.get('type') != 'fixed':
                active_joints.append(i.name)
        arm_joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']

        print('len(active_joints):', len(active_joints))
        self.register_controller(
            'arm', PDJointPosController(
                PDJointPosConfig(
                    lower=[-3.14],
                    upper=[3.14],
                    stiffness=200.,
                    damping=50.,
                    force_limit=100.,
                ), 
                self, arm_joints
            )
        )

        gripper_joints = ['left_finger_joint', 'right_finger_joint']
        self.register_controller(
            "gripper", MimicPDJointPosController(PDJointPosConfig(
                lower=0.,  # a trick to have force when the object is thin
                upper=0.85,
                stiffness=20,
                damping=5,
                force_limit=10.,

            ), self, gripper_joints)
        )

    def get_ros_plugins(self):
        from ..ros_plugins.ros_plugin import ArmServo
        return [ArmServo(self.controllers['arm'])]

    @property
    def ee_name(self):
        return 'xarm_gripper_base_link'

    @property
    def base_name(self):
        return 'link_base'
    
    def _post_load(self, world: Simulator):
        #def add_loop_for_gripper(robot: PhysxArticulation, scene: Scene, pmodel=None):
        #if pmodel is None:
        robot = self.articulation
        robot.set_qpos(np.zeros(robot.dof, dtype=np.float32))
        scene = world._scene
        import sapien
        pmodel = robot.create_pinocchio_model() # type: ignore
        pmodel.compute_forward_kinematics(np.zeros(robot.dof))

        # robot.set_qpos(np.zeros(robot.dof)) # type: ignore
        def get_pose(link: sapien.physx.PhysxArticulationLinkComponent):
            return pmodel.get_link_pose(link.index)

        flange = robot.find_link_by_name("joint6_flange")

        lik = robot.find_link_by_name("left_inner_knuckle")
        lok = robot.find_link_by_name("left_outer_knuckle")
        lf = robot.find_link_by_name("left_finger")
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

        rik = robot.find_link_by_name("right_inner_knuckle")
        rok = robot.find_link_by_name("right_outer_knuckle")
        rf = robot.find_link_by_name("right_finger")
        T_pw = get_pose(rf).inv().to_transformation_matrix()
        p_w = get_pose(rf).p + (get_pose(rik).p - get_pose(rok).p) # type: ignore
        T_fw = get_pose(rik).inv().to_transformation_matrix()
        p_f = T_fw[:3, :3] @ p_w + T_fw[:3, 3]
        p_p = T_pw[:3, :3] @ p_w + T_pw[:3, 3]
        drive = scene.create_drive(rik, Pose(p_f), rf, Pose(p_p))
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)
        #raise NotImplementedError

        # gear = scene.create_gear(lok, Pose(), rok, Pose(q=[0, 0, 0, 1]))
        # gear.gear_ratio = -1
        # gear.enable_hinges()

        for idx, l in enumerate([lik, lok, lf, rik, rok, rf]):
            if l is not None:
                for s in l.collision_shapes:
                    s.set_collision_groups([1, 1, 4, 0])
        return [j.name for j in [lik, lok, lf, rik, rok, rf] if j is not None]
