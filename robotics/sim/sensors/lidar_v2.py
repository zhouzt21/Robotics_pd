# a fake lidar class
import sapien.render
import copy
from typing import List, Union, Dict, TypedDict

import sapien
from sapien.render import RenderCameraComponent
from sapien import physx
from .sensor_base import SensorBase, Config, SensorConfig, Pose
from ..simulator import Simulator
from .camera_v2 import CameraConfig, CameraV2
import transforms3d
import numpy as np

from robotics.utils.path import PACKAGE_DIR

class LidarConfig(CameraConfig):
    pass

class LidarObservation(TypedDict):
    points : np.ndarray


class Lidar(SensorBase[LidarConfig]):
    def __init__(self, sensor_cfg: LidarConfig) -> None:
        assert sensor_cfg.look_at is None, "Lidar cannot look at anything"
        super().__init__(sensor_cfg)

    def _load(self, world: Simulator):
        """_summary_
        we just need to load all the subentities
        """
        shader_path = PACKAGE_DIR / 'shaders' / 'lidar'
        sapien.render.set_camera_shader_dir(str(shader_path))

        actor, relative_pose = self.get_frame(world)

        # Divide 360 degree into 64*64 rays, how rows and cols are distributed is not important
        lidar_camera = RenderCameraComponent(self.config.width, self.config.width) 
        # set near in lidar/camera.rgen/tmin

        if actor is None:
            actor = sapien.Entity()
            world._scene.add_entity(actor)
            actor.set_pose(Pose())

        self.mount = actor
        self.camera = lidar_camera
        self.mount.add_component(lidar_camera)
        self.camera.set_local_pose(relative_pose)
        self.world = world


        sapien.render.set_camera_shader_dir(world._default_shader_dir)
    

    def get_observation(self) -> LidarObservation:
        #TODO: It is better we can override the get_observation function
        #   the code below returns the points in the world frame
        self.world.update_scene_if_needed()

        self.camera.take_picture()
        position2 = self.camera.get_picture("Position")
        # print(position2)
        points = position2.reshape(-1, 4)
        points = points[points[..., 3] > 0]
        #points = points[..., [0, 2, 1, 3]] # the original order is x, z, y, depth.
        points[..., 3] = 1

        camera_matrix = self.camera.get_model_matrix()
        points = points @ camera_matrix.T

        return {'points': points}

    def get_pose(self):
        return self.mount.get_pose() * self.camera.get_local_pose() if self.mount is not None else self.camera.get_pose()