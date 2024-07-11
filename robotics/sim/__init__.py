"""
## Simulator Module

- simulator.py: group of entities that can be simulated together
- entity.py: The basic element of the simulated environment. Everything can be an entity and an entity can have a sub-entity to form a tree structure.
- robot: a robot class that contains:
    - A sapien articulated object, reading from the URDF
    - sensors mounted at the robot, robot proprioception, lidar and RGBD sensors
    - controllers
- sensors: camera, lidar and so on
- environs: scenes and objects that can be composed into the environment
"""
from .engines.gpu_engine import GPUEngineConfig, GPUEngine, ensure_rigid_comp
from .simulator import Simulator, SimulatorConfig
from .environ import EnvironBase
from .sensors.camera_v2 import CameraV2, CameraConfig
from .sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig


from .robot.robot_base import Robot as RobotBase
from .robot.mobile_small_v2 import MobileSmallV2 as MobileSmallRobot
from .simulator_base import get_engine, CPUEngine, GPUEngine