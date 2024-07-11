from robotics.cfg import Config
from robotics import Pose
import sapien.core as sapien
from typing import Dict, Union, List, TYPE_CHECKING, Optional, Sequence
from dataclasses import dataclass, field
from .entity import Entity


class EnvironConfig(Config):
    p: tuple[float, ...] = (0, 0, 0)
    q: tuple[float, ...] = (1., 0, 0, 0)


class EnvironBase(Entity):
    def __init__(self, config: EnvironConfig):
        self.config = config

    def _get_sapien_entity(self) -> List[Union[sapien.Entity, sapien.physx.PhysxArticulation]]:
        raise NotImplementedError(f"_get_sapien_entity of class {self.__class__.__name__} is not implemented")

    def _get_config_pose(self):
        return Pose(self.config.p, self.config.q)