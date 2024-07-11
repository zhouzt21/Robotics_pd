"""_summary_
Maniskill2 styple controller but simplified
"""

import numpy as np
from typing import List, Union, Dict, Any, TYPE_CHECKING, cast, Sequence, Literal, Optional
import torch
from torch import Tensor
from functools import cached_property


from robotics.cfg import Config
from gymnasium import spaces

from sapien import physx
from ..entity import Entity

if TYPE_CHECKING:
    from ..simulator import Simulator
    from .robot_base import Robot


class ControllerConfig(Config):
    lower: Union[float, List[float]] = -1
    upper: Union[float, List[float]] = 1
    normalize_action: bool = True

    damping: Union[float, List[float]] = 0.0
    force_limit: Union[float, List[float]] = 1e10
    friction: Union[float, List[float]] = 0.0
    stiffness: Union[float, List[float]] = 0.0
    mode: Literal["force", "acceleration"] = "force"



class Controller(Entity):
    def __init__(
        self, 
        config: ControllerConfig,
        robot: "Robot",
        joint_names_or_ids: Union[List[str], List[int]],
    ) -> None:
        super().__init__()
        self.config = config
        self.robot = robot
        self.joint_names_or_ids = joint_names_or_ids

    def _clear(self):
        setattr(self, "articulation", None)
        setattr(self, "joint_indices", None)
        setattr(self, "joints", None)
        setattr(self, "_action_space", None)

    def get_state(self) -> Optional[np.ndarray]:
        return None

    def set_drive_targets(self, targets):
        self.robot.engine.set_joint_target(self.articulation, self.joints, self.joint_indices, targets)

    def set_drive_velocity_targets(self, targets):
        self.robot.engine.set_joint_velocity_target(
            self.articulation, self.joints, self.joint_indices, targets
        )


    def set_state(self, state: np.ndarray):
        raise NotImplementedError(f"Cannot set state of controller {type(self)}")

    def inv_scale_action(self, action):
        """Inverse of `clip_and_scale_action` without clipping."""
        low, high = self.get_low, self.get_high
        return (action - 0.5 * (high + low)) / (0.5 * (high - low))


    def _load(self, world: "Simulator"):
        assert world.is_loaded(self.robot), f"Robot {self.robot} is not loaded"
        self.articulation = self.robot.articulation
        self.joint_indices, self.joints = self.robot.get_active_joints(self.joint_names_or_ids)
        self._action_space = None
        self.n_scenes = world.config.n_scenes
        self._sim_freq = world._sim_freq

        self.device = world._engine.device 

        self.host = np if self.device == 'numpy' else torch
        self.to_host = np.asarray if self.device == 'numpy' else lambda x: torch.tensor(x).float().to(self.device)

        self.reset()

    def reset(self):
        raise NotImplementedError(f"Cannot reset {type(self)}")

    def get_action_space(self):
        raise NotImplementedError(f"Cannot get action space of {type(self)}")

    def _get_sapien_entity(self) -> List[Union[Entity, physx.PhysxArticulation]]:
        return []

    @property
    def qpos(self):
        """Get current joint positions."""
        #return self.articulation.get_qpos()[self.joint_indices]
        return self.robot.engine.get_qpos(self.articulation)[..., self.joint_indices]

    @property
    def qvel(self):
        """Get current joint velocities."""
        return self.robot.engine.get_qvel(self.articulation)[..., self.joint_indices]
        
    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = self.get_action_space()
        return self._action_space

    @cached_property
    def get_low(self):
        #return self.config.lower
        return self.to_host(self.config.lower)
    
    @cached_property
    def get_high(self):
        #return self.config.upper
        return self.to_host(self.config.upper)

    def clip_and_scale_action(self, action, low, high) -> np.ndarray | Tensor:
        """Clip action to [-1, 1] and scale according to a range [low, high]."""
        if self.device == 'numpy':
            action = np.clip(action, -1, 1)
        else:
            action = torch.clamp(action, -1, 1)
        return 0.5 * (high + low) + 0.5 * (high - low) * action


    def _preprocess_action(self, action: np.ndarray | Tensor):
        # TODO(jigu): support discrete action
        if self.device == 'numpy':
            if isinstance(action, Tensor):
                action = action.cpu().numpy()
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action).float().to(self.device)

        action_dim = self.action_space.shape[-1]
        if self.n_scenes > 1:
            assert action.shape == (self.n_scenes, action_dim), (action.shape, (self.n_scenes, action_dim))
        else:
            assert action.shape == (action_dim,), (action.shape, action_dim)
        if self.config.normalize_action:
            action =self.clip_and_scale_action(action, self.get_low, self.get_high)
        return action

    @property
    def _control_freq(self):
        return self.robot.control_freq

    def drive_target(self, target):
        raise NotImplementedError(f"Cannot drive to {type(self)}")

    def set_action(self, action: Union[np.ndarray, Sequence[float]]):
        raise NotImplementedError


class PDJointVelConfig(ControllerConfig):
    pass
        
class PDJointVelController(Controller):
    config: PDJointVelConfig
    def get_action_space(self):
        n = len(self.joints)
        low = np.float32(np.broadcast_to(self.config.lower, n))
        high = np.float32(np.broadcast_to(self.config.upper, n))
        return spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(stiffness[i], damping[i], force_limit=force_limit[i], mode=self.config.mode)
            joint.set_friction(friction[i])

    def set_action(self, action: np.ndarray | Tensor):
        action = self._preprocess_action(action)
        # for i, joint in enumerate(self.joints):
        #     joint.set_drive_velocity_target(action[i])
        self.robot.engine.set_joint_velocity_target(self.articulation, self.joints, self.joint_indices, action)
        
    def _action_from_delta_qpos(self, delta_qpos):
        return delta_qpos

    def drive_target(self, target):
        # assert isinstance(controller, PDJointVelController), type(controller)
        assert self.config.normalize_action
        delta_qpos = target - self.qpos
        qvel = delta_qpos * self._control_freq
        qvel = self._action_from_delta_qpos(qvel)
        return self.inv_scale_action(qvel)


            


class PDJointPosConfig(ControllerConfig):
    mode: Literal["force", "acceleration"] = "force"


class PDJointPosController(Controller):
    config: PDJointPosConfig
    def _get_joint_limits(self):
        qlimits = self.robot.articulation.get_qlimit()[self.joint_indices]
        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return cast(np.ndarray, qlimits)

    def get_action_space(self):
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        return spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(
                stiffness[i], damping[i], force_limit=force_limit[i], mode=self.config.mode
            )
            joint.set_friction(friction[i])


    def set_action(self, action: np.ndarray | Tensor):
        action = self._preprocess_action(action)
        self._start_qpos = self.qpos
        # print(action.shape, 'start qpos', self._start_qpos.shape)
        self._target_qpos = self.host.broadcast_to(action, self._start_qpos.shape) # type: ignore
        self.set_drive_targets(self._target_qpos)
        

    def drive_target(self, target):
        return self.inv_scale_action(target)


class MimicPDJointPosController(PDJointPosController):
    def _get_joint_limits(self):
        assert len(self.joints) == 2, "Mimic joints should have at least 2 joints"
        joint_limits = super()._get_joint_limits()
        diff = joint_limits[0:-1] - joint_limits[1:]
        assert np.allclose(diff, 0), "Mimic joints should have the same limit"
        return joint_limits[0:1]