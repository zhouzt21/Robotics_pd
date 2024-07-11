# ee_delta _controller 
import transforms3d
import numpy as np
from gymnasium import spaces
import sapien
from .controller import PDJointPosController, PDJointPosConfig
from typing import TYPE_CHECKING, Any
from robotics.sim import get_engine

if TYPE_CHECKING:
    from ..simulator import Simulator
    from .robot_base import Robot


class EEDeltaPoseConfig(PDJointPosConfig):
    ik_iterations: int = 100
    lower: tuple[float, ...] = (-0.5, -0.5, -0.5, -1, -1, -1)
    upper: tuple[float, ...] = (0.5, 0.5, 0.5, 1, 1, 1)
    use_target: bool = False

    #ik_bounds: tuple[tuple[float, float], ...] = ((-0.3, 0.3), (0., 0.5), (0.05, 0.3))
    ik_lower: tuple[float, ...] = (0.0, -0.3, 0.05)
    ik_higher: tuple[float, ...] = (0.5, 0.3, 0.3)


class EEDeltaPoseController(PDJointPosController):
    config: EEDeltaPoseConfig

    def __init__(
        self, 
        config: EEDeltaPoseConfig,
        robot: "Robot",
        joint_names_or_ids: list[str] | list[int],
        ee_name: str,
        frame: str,
    ) -> None:
        super().__init__(config, robot, joint_names_or_ids)
        self.ee_name = ee_name
        self.frame_name = frame
    
    def get_state(self) -> dict[str, Any] | None:
        return {
            'target_pose': np.concatenate((self.target_pose.p, self.target_pose.q), axis=-1),
            'ik_qpos': self.ik_qpos,
        }

    def set_state(self, state):
        self.ik_qpos = state['ik_qpos']
        self.target_pose = sapien.Pose(state['target_pose'][:3], state['target_pose'][3:])

    def _load(self, sim: "Simulator"):
        self.ee_link = self.robot.articulation.find_link_by_name(self.ee_name)
        self.base_link = self.robot.articulation.find_link_by_name(self.frame_name)

        self.block_material = sim._renderer.create_material()
        builder = sim._scene.create_actor_builder()
        self.block_material.base_color = [1., 0., 0., 1.]
        builder.add_box_visual(sapien.Pose((0, 0, 0.)), (0.02, 0.02, 0.02), self.block_material)
        self.block = builder.build_kinematic()
        #self.ik = self.robot.ik


        super()._load(sim)
        self.engine = get_engine()
        assert isinstance(self.engine, sapien.Engine)
        self.pmodel = self.robot.pmodel
        self.qmask = np.zeros(
            self.robot.articulation.dof
        )
        self.qmask[self.joint_indices] = 1
        self.dt = sim.control_timestep

        self.ik_qpos = self.qpos

    def reset(self):
        super().reset()
        self.target_pose = self.base_link.get_pose().inv() * self.ee_link.get_pose()
        self.block.set_pose(self.base_link.get_pose() * self.target_pose)

    def get_action_space(self):
        return spaces.Box(
            low=-1,
            high=1,
            shape=(6,),
            dtype=np.float32,
        ) 


    def ik(self, pose: sapien.Pose):
        result, success, error = self.pmodel.compute_inverse_kinematics(
            self.ee_link.index,
            pose,
            initial_qpos=self.articulation.get_qpos(),
            active_qmask=self.qmask,
            max_iterations=self.config.ik_iterations,
        )

        ik0 = result[self.joint_indices]

        for j in range(len(self.joint_indices)):
            val = ik0[j]
            while val > np.pi:
                val = val - np.pi * 2
            while val < -np.pi:
                val = val + np.pi * 2
            ik0[j] = val

        # from loguru import logger
        # logger.debug(f"ik result: {'success' if success else 'fail'}")
        if success:
            return ik0, success
        else:
            # print(result, error)
            return None, False

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        self.cur_base_pose = self.base_link.get_entity_pose()
        if self.config.use_target:
            self.cur_relative_pose = self.target_pose
        else:
            self.cur_pose = self.ee_link.get_entity_pose()
            self.cur_relative_pose = self.cur_base_pose.inv() * self.cur_pose

            action = action * 10

        new_xyz = self.cur_relative_pose.p + action[:3] * self.dt
        #new_rot = sapien.Quaternion.from_xyzw(action[3:]) * self.cur_relative_pose.q

        action_rot = action[3:] * self.dt
        norm = np.linalg.norm(action_rot)
        if norm < 1e-6:
            new_rot = self.cur_relative_pose.q
        else:
            quat = transforms3d.quaternions.axangle2quat(action_rot/norm, norm)
            new_rot = transforms3d.quaternions.qmult(quat, self.cur_relative_pose.q)
            new_rot = new_rot / np.linalg.norm(new_rot)

        new_xyz = np.clip(new_xyz, self.config.ik_lower, self.config.ik_higher)
        target_pose = sapien.Pose(new_xyz, new_rot)

        #self.set_drive_targets(self._target_qpos)
        target_world_frame = self.cur_base_pose * target_pose
        ik, success = self.ik(target_world_frame)

        if success:
            self.ik_qpos = ik
            self.target_pose = target_pose

        self.block.set_pose(self.cur_base_pose * self.target_pose)
        self.set_drive_targets(self.ik_qpos)