from typing import Sequence, Union
import torch
from torch import Tensor
from .controller import Controller, ControllerConfig, spaces, PDJointVelController
import numpy as np

def rotate_2d_vec_by_angle(vec, theta):
    if isinstance(theta, np.ndarray):
        ct = np.cos(theta)
        st = np.sin(theta)
    else:
        ct = torch.cos(theta)
        st = torch.sin(theta)

    x = vec[..., 0] * ct - vec[..., 1] * st
    y = vec[..., 0] * st + vec[..., 1] * ct

    if isinstance(theta, np.ndarray):
        return np.stack([x, y], axis=-1)
    else:
        return torch.stack([x, y], dim=-1)

    
    
class PDBaseVelController(PDJointVelController):
    def set_action(self, action: np.ndarray | Tensor):
        action = self._preprocess_action(action)
        # Convert to ego-centric action
        # Assume the 3rd DoF stands for orientation

        ori = self.qpos[..., 2]
        vel = rotate_2d_vec_by_angle(action[..., :2], ori)
        if isinstance(vel, np.ndarray):
            new_action = np.concatenate([vel, action[..., 2:]], axis=-1)
        else:
            new_action = torch.cat([vel, action[..., 2:]], dim=-1) # type: ignore
        # new_action = tu.concat([vel, action[..., 2:]], dim=-1)

        # print('drive to', new_action)
        # for i, joint in enumerate(self.joints):
        #     joint.set_drive_velocity_target(new_action[i])
        self.robot.engine.set_joint_velocity_target(self.articulation, self.joints, self.joint_indices, new_action)

    def _action_from_delta_qpos(self, delta_qpos):
        assert len(delta_qpos.shape) == 1
        ori = self.qpos[2]
        delta_qpos[:2] = rotate_2d_vec_by_angle(delta_qpos[:2], -ori) # inv it..
        delta_qpos[2] = (delta_qpos[2] + np.pi) % (2 * np.pi) - np.pi
        return delta_qpos

    


class DifferentialDriveControllerConfig(ControllerConfig):
    damping: Union[float, Sequence[float]]  = 0.0
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0


class DifferentialDriveController(Controller):
    config: "DifferentialDriveControllerConfig"

    def _initialize_action_space(self):
        assert len(self.joints) == 3
        n = 2
        low = np.float32(np.broadcast_to(self.config.lower, n))
        high = np.float32(np.broadcast_to(self.config.upper, n))
        return spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        n = len(self.joints)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(0, damping[i], force_limit=force_limit[i])
            joint.set_friction(friction[i])

    def set_action(self, action: np.ndarray):

        action = self._preprocess_action(action)

        # Convert to ego-centric action
        # Assume the 3rd DoF stands for orientation
        raise NotImplementedError("Not implemented yet. Should be similar to PDBaseVelController")
        ori = self.qpos[2]
        vel = rotate_2d_vec_by_angle(np.array([action[0], 0]), ori)
        new_action = np.hstack([vel, action[1]])

        # for i, joint in enumerate(self.joints):
        #     joint.set_drive_velocity_target(new_action[i])
        self.robot.engine.set_joint_velocity_target(self.articulation, self.joints, self.joint_indices, new_action)

    def drive_target(self, target):
        raise NotImplementedError
        controller = self.base_controller
        delta_qpos = qpos[:3] - controller.qpos
        qvel = delta_qpos * controller._control_freq
        ori = controller.qpos[2]
        qvel[:2] = rotate_2d_vec_by_angle(qvel[:2], -ori) # inv it..
        qvel1 = np.array([qvel[0], qvel[2]])
        qvel1 = inv_scale_action(qvel1, controller.config.lower, controller.config.upper)
