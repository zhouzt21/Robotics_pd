from dataclasses import dataclass, field
import numpy as np
from robotics.cfg import Config
from typing import Optional, Union, Tuple

Vec3 = Tuple[float, float, float]

class SensorConfig(Config):
    """_summary_
    sensor config describes the initial relative pose towards the base entity 
    look at will define the initial orientation of the sensor
    """
    base: str = "world"
    look_at: Optional[Tuple[float, float, float]] = field(default_factory=lambda : None)

    p: Tuple[float, float, float] = (0, 0, 0)
    q: Tuple[float, float, float, float] = (1, 0, 0, 0) # wxyz
    # pose: Pose_ = field(default_factory=Pose_)

class CameraConfig(SensorConfig):
    # type: CameraType = CameraType.camera
    width: int = 512
    height: int = 512
    fov: float = 1.0
    near: float = 0.1
    far: float = 100.
    uid: str = "camera"

    hide_link: bool = False
