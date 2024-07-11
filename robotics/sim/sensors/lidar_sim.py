# a fake lidar class
import copy
from typing import List, Union, Dict

import sapien
from sapien import physx
from .sensor_base import SensorBase, Config, SensorConfig, Pose, Simulator
from .camera_v2 import CameraConfig, CameraV2
import transforms3d
import numpy as np

class LidarConfig(CameraConfig):
    pass

class Lidar(SensorBase[LidarConfig]):
    def __init__(self, sensor_cfg: LidarConfig) -> None:
        assert sensor_cfg.look_at is None, "Lidar cannot look at anything"
        super().__init__(sensor_cfg)

        cameras: Dict[str, CameraV2] = {}

        for idx, theta in enumerate([0., np.pi/2, np.pi, np.pi * 3/2]):
            q = np.array([np.cos(theta/2), 0., 0., np.sin(theta/2)])
            p = transforms3d.quaternions.rotate_vector(np.array([0.3, 0., 0.1]), q)

            cam_cfg = copy.deepcopy(self.config)
            cam_cfg.p = p.tolist()
            cam_cfg.q = q.tolist()

            cameras[f'lidar_{idx}'] = CameraV2(cam_cfg)
            self.add_subentity(f'{idx}', cameras[f'lidar_{idx}'])
        self.cameras = cameras
        # self.mount = self.cameras['lidar_0'].mount
        # assert self.mount is not None


    def _load(self, world: Simulator):
        """_summary_
        we just need to load all the subentities
        """
        self.mount = None

    
    def _get_observation(self):
        #TODO: It is better we can override the get_observation function
        return {}

    def get_pose(self):
        if self.mount is None:
            mount = self.cameras['lidar_0'].mount
            assert mount is not None
            self.mount = mount
        return self.mount.get_pose()