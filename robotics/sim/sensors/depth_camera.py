from .camera_v2 import CameraConfig, CameraV2, Simulator
from loguru import logger
import numpy as np

from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
from typing import cast

from robotics import Pose
from sapien.render import (
    RenderCameraComponent,
    RenderTexturedLightComponent,
    RenderTexture2D,
)



class StereoDepthCameraConfig(CameraConfig):
    min_depth: float = 0.05
    camera_type: str = "D415"
    width: int = 1980
    height: int = 1080
    fov: float = 0.749

    @property
    def rgb_resolution(self):
        return (self.width, self.height)

    @property
    def rgb_intrinsic(self):
        fy = (self.height / 2) / np.tan(self.fov / 2)
        return np.array([[fy, 0, self.width / 2], [0, fy, self.height / 2], [0, 0, 1]])

        
class _StereoDepthSensor(StereoDepthSensor):
    def _create_cameras(self):
        self._cam_rgb = RenderCameraComponent(*self._config.rgb_resolution)
        self._cam_rgb.local_pose = self._pose
        self._cam_rgb.name = "cam_rgb"
        near = 0.1
        self._cam_rgb.set_perspective_parameters(
            near,
            100.0,
            self._config.rgb_intrinsic[0, 0],
            self._config.rgb_intrinsic[1, 1],
            self._config.rgb_intrinsic[0, 2],
            self._config.rgb_intrinsic[1, 2],
            self._config.rgb_intrinsic[0, 1],
        )

        self._cam_ir_l = RenderCameraComponent(*self._config.ir_resolution)
        self._cam_ir_l.local_pose = self._pose * self._config.trans_pose_l
        self._cam_ir_l.name = "cam_ir_l"
        self._cam_ir_l.set_perspective_parameters(
            near,
            100.0,
            self._config.ir_intrinsic[0, 0],
            self._config.ir_intrinsic[1, 1],
            self._config.ir_intrinsic[0, 2],
            self._config.ir_intrinsic[1, 2],
            self._config.ir_intrinsic[0, 1],
        )

        self._cam_ir_r = RenderCameraComponent(*self._config.ir_resolution)
        self._cam_ir_r.local_pose = self._pose * self._config.trans_pose_r
        self._cam_ir_r.name = "cam_ir_r"
        self._cam_ir_r.set_perspective_parameters(
            near,
            100.0,
            self._config.ir_intrinsic[0, 0],
            self._config.ir_intrinsic[1, 1],
            self._config.ir_intrinsic[0, 2],
            self._config.ir_intrinsic[1, 2],
            self._config.ir_intrinsic[0, 1],
        )

        self._mount.add_component(self._cam_rgb)
        self._mount.add_component(self._cam_ir_l)
        self._mount.add_component(self._cam_ir_r)

        self._cam_ir_l.set_property("exposure", float(self._config.ir_camera_exposure))
        self._cam_ir_r.set_property("exposure", float(self._config.ir_camera_exposure))



def D435_matrix():
    k_irl = k_irr = np.array([[430.13980103,   0.        , 425.1628418 ],
                            [  0.        , 430.13980103, 235.27651978],
                            [  0.        ,   0.        ,   1.        ]])

    # rsdevice.all_extrinsics["Infrared 1=>Infrared 2"]
    T_irr_irl = np.array([[ 1.       ,  0.       ,  0.       , -0.0501572],
                        [ 0.       ,  1.       ,  0.       ,  0.       ],
                        [ 0.       ,  0.       ,  1.       ,  0.       ],
                        [ 0.       ,  0.       ,  0.       ,  1.       ]])
    # rsdevice.all_extrinsics["Infrared 1=>Color"]
    T_rgb_irl = np.array([[ 9.99862015e-01,  1.34780351e-02,  9.70994867e-03, 1.48976548e-02],
                        [-1.35059441e-02,  9.99904811e-01,  2.81448336e-03, 1.15314942e-05],
                        [-9.67109110e-03, -2.94523709e-03,  9.99948919e-01, 1.56505470e-04],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
    return k_irl, T_irr_irl, T_rgb_irl


class StereoDepthCamera(CameraV2[StereoDepthCameraConfig]):
    def _load(self, world: Simulator):
        super()._load(world)

        sensor_config = StereoDepthSensorConfig()

        if self.config.camera_type == "D415":
            sensor_config.rgb_resolution = self.config.rgb_resolution # type: ignore
            sensor_config.rgb_intrinsic = self.config.rgb_intrinsic
        elif self.config.camera_type == "D435":
            if self.config.rgb_resolution != (848, 480):
                logger.warning("D435 only support 848x480 resolution")
            k_irl, T_irr_irl, T_rgb_irl = D435_matrix()

            T_irl_rgb = np.linalg.inv(T_rgb_irl)
            T_irr_rgb = T_irr_irl @ T_irl_rgb

            sensor_config.rgb_resolution = self.config.rgb_resolution # type: ignore
            sensor_config.rgb_intrinsic = np.array(
                [[604.92340088,   0.,         424.50192261],
                [  0.,         604.59521484, 246.3170166 ],
                [  0.,           0.,           1.        ]]
            )
            print('T_irl_rgb', T_irl_rgb)
            print('T_irr_rgb', T_irr_rgb)
            sensor_config.trans_pose_l = Pose(np.array(
                [[0.9999489188194275, 0.009671091102063656, 0.0029452370945364237, 0.00015650546993128955],
                [-0.009709948673844337, 0.999862015247345, 0.013478035107254982, -0.014897654764354229],
                [-0.0028144833631813526, -0.013505944050848484, 0.9999048113822937, -1.1531494237715378e-05],
                [0.0, 0.0, 0.0, 1.0]]
            ))
            sensor_config.trans_pose_r = Pose(np.array(
                [[0.9999489188194275, 0.009671091102063656, 0.0029452370945364237, -0.0003285693528596312],
                [-0.009709948673844337, 0.999862015247345, 0.013478035107254982, -0.06504792720079422],
                [-0.0028144833631813526, -0.013505944050848484, 0.9999048113822937, 0.0006658887723460793],
                [0.0, 0.0, 0.0, 1.0]]
                )
            )
            sensor_config.ir_intrinsic = k_irl
        else:
            raise NotImplementedError

        

        sensor_config.min_depth = self.config.min_depth

    
        actor, relative_pose = self.get_frame(world)
        # print(actor.get_pose(), relative_pose)

        # scene = world._scene
        #assert actor is not None or relative_pose is None
        self._fake_actor = None
        if actor is None:
            actor = world._scene.create_actor_builder().build_static(name='fake_camera_root')
            self._fake_actor = actor
            self.relative_pose = relative_pose

        mount = {
            'mount_entity': actor,
            'pose': relative_pose
        }
        self.relative_pose = relative_pose

        self.camera = _StereoDepthSensor(
            sensor_config, **mount
        )
        self._cam_rgb = self.camera._cam_rgb
        self.scene = world._scene
        if self.config.hide_link:
            #TODO: just a hack to hide the link
            world._show_camera_linesets = False
            for cam in [self._cam_rgb, self.camera._cam_ir_l, self.camera._cam_ir_r]:
                pass
        print('_cam_rgb pose', self._cam_rgb.get_pose())

    def get_pose(self):
        #assert self._fake_actor is not None, "Cannot get pose for non-fake camera.."
        return self.camera.get_pose()

    def set_pose(self, pose):
        assert self._fake_actor is not None, "Cannot set pose for non-fake camera.."
        pose = pose * self.relative_pose.inv()
        self._fake_actor.set_pose(pose)

    def get_observation(self, gt_depth: bool=False):
        """Get (raw) images from the camera."""
        # TEXTURE_DTYPE = {"Color": "float", "Position": "float", "Segmentation": "uint32"}
        self.world.update_scene_if_needed()

        images = {}
        self._cam_rgb.take_picture() # type: ignore
        for name in ['Color', 'Segmentation']:
            images[name] = self._cam_rgb.get_picture(name)
        if gt_depth:
            images['depth_gt'] = -self._cam_rgb.get_picture("Position")[..., 2]

        # position = self._cam_rgb.get_picture("Position")

        self.camera._ir_mode()
        self.scene.update_render()
        self.camera._cam_ir_l.take_picture() # type: ignore
        self.camera._cam_ir_r.take_picture() # type: ignore

        ir_l, ir_r = self.camera.get_ir()
        self.camera.compute_depth()
        depth = self.camera.get_depth()

        self.camera._normal_mode()
        self.scene.update_render()

        images['depth'] = depth
        images['ir_l'] = ir_l
        images['ir_r'] = ir_r
        return images

    def get_camera_params(self):
        """Get camera parameters."""
        assert self._cam_rgb is not None
        return dict(
            extrinsic_cv=self._cam_rgb.get_extrinsic_matrix(),
            cam2world_gl=self._cam_rgb.get_model_matrix(),
            intrinsic_cv=self._cam_rgb.get_intrinsic_matrix(),
        )

        
    def get_point_cloud(self, data):
        #TODO: verify it
        assert self._cam_rgb is not None
        intrinsic = cast(np.ndarray, self._cam_rgb.get_intrinsic_matrix())
        extrinsic = self.get_pose().inv().to_transformation_matrix() @ self._cam_rgb.get_model_matrix() # type: ignore

        
        depth = data[:, :, 3]
        width, height = depth.shape[1], depth.shape[0]

        coords = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
        coords = coords[:,::-1] + 0.5

        coords = np.concatenate((coords, np.ones((height, width, 1))), axis=-1)
        coords = coords @ np.linalg.inv(intrinsic).T

        position = coords * depth[..., None]
        rgb = data[:, :, :3]

        position = (position @ extrinsic[:3, :3].T + extrinsic[:3, 3])
        return position.reshape(-1, 3), rgb.reshape(-1, 3)/255.

    def _get_sapien_entity(self):
        return [self.camera, self._fake_actor] if self._fake_actor is not None else [self.camera]