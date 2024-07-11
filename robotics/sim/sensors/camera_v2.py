import sapien
from sapien import physx
from robotics.sim.simulator import Simulator
from .sensor_base import SensorBase, Config
from .sensor_cfg import CameraConfig
from robotics import Pose 
from typing import Any, List, Union, TypeVar
from robotics.utils.sapien_utils import get_render_body_component

CC = TypeVar('CC', bound=CameraConfig)



class CameraV2(SensorBase[CC]):
    TEXTURE_DTYPE = {"Color": "float", "Position": "float", "Segmentation": "uint32"}

    def _load(self, world: Simulator):
        actor, relative_pose = self.get_frame(world)
        scene = world._scene

        camera_cfg_ = self.config
        camera_cfg: dict[str, Any] = dict(
            width=camera_cfg_.width,
            height=camera_cfg_.height,
            fovy=camera_cfg_.fov,
            near=camera_cfg_.near,
            far=camera_cfg_.far,
        )

        if actor is None:
            self.mount = None
            self.camera = scene.add_camera(self.config.uid, **camera_cfg)
            self.camera.set_local_pose(relative_pose)
        else:
            self.mount = actor
            self.camera = scene.add_mounted_camera(self.config.uid, actor, relative_pose, **camera_cfg)

        if self.config.hide_link:
            #TODO: just a hack to hide the link
            world._show_camera_linesets = False

        self.world = world



    def _get_sapien_entity(self):
        return [self.camera]


    def get_observation(self):
        """Get (raw) images from the camera."""
        self.world.update_scene_if_needed()

        self.camera.take_picture()
        images = {}
        for name in self.TEXTURE_DTYPE:
            image = self.camera.get_picture(name)
            images[name] = image

        images['depth'] = -images.pop("Position")[..., 2]
        return images


    def get_camera_params(self):
        # extrinsic = np.linalg.inv(tf.to_transformation_matrix())  @ camera_params[camera_name]['cam2world_gl']
        params = dict(
            extrinsic_cv=self.camera.get_extrinsic_matrix(),
            cam2world_gl=self.camera.get_model_matrix(),
            intrinsic_cv=self.camera.get_intrinsic_matrix(),
        )

        return {
            **params,
            "height": self.config.height,
            "width": self.config.width,
            "fov": self.config.fov,
            "near": self.config.near,
            "far": self.config.far,
            **params,
        }

    def get_pose(self) -> Pose:
        #if self.mount is not None:
        #    print(self.mount.get_pose().p - self.camera.get_pose().p, self.camera.local_pose)
        from typing import cast
        return cast(Pose, self.camera.get_pose() * self.camera.local_pose)

    def set_pose(self, pose: Pose):
        #self.camera.set_pose(pose)
        if self.mount is not None:
            self.mount.set_pose(pose * self.camera.local_pose.inv())
        else:
            self.camera.set_local_pose(pose)