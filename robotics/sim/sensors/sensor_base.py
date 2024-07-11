"""_summary_
Base class for sensors. 
A sensor is attached to a frame, which can be either the world frame or an entity frame.
Currently we require the entity to be an entity in sapien3.
"""
import sapien
from ..entity import Entity
from robotics.cfg import Config
from typing import List, Union, Dict, Any, TYPE_CHECKING, cast, Sequence, Optional, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from robotics.utils.sapien_utils import look_at
from robotics import Pose
import numpy as np

from .sensor_cfg import SensorConfig
from ..simulator import Simulator




def get_pose_from_sensor_cfg(config: SensorConfig):
    p, q = config.p, config.q
    if config.look_at is not None:
        q = look_at(p, config.look_at).q.tolist()
    return Pose(p, q)


SC = TypeVar('SC', bound=SensorConfig)

class SensorBase(Entity, Generic[SC]):
    config: SC
    def __init__(
        self, 
        sensor_cfg: SC,
    ) -> None:
        super().__init__()
        self.config = sensor_cfg

    def get_frame(self, world: "Simulator")->Tuple[Optional[sapien.Entity], Pose]:
        if self.config.base == 'world': 
            base_entity = None
        else:
            base_entity = world.find(self.config.base)

        if base_entity is not None:
            if not isinstance(base_entity, sapien.Entity):
                # NOTE: extract the entity from the sapien links
                base_entity = getattr(base_entity, 'entity')
        return base_entity, get_pose_from_sensor_cfg(self.config)

    def get_params(self):
        # return the parameters of the sensor
        raise NotImplementedError(f"Cannot get params of {type(self)}")

    def get_pose(self):
        # return the pose of the sensor
        raise NotImplementedError(f"Cannot get pose of {type(self)}")

    def _get_observation(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError(f"Cannot get observation of {type(self)}")