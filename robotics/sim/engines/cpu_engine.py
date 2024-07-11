from sapien import Engine, Entity, Scene, Pose
import torch
from torch import Tensor
import sapien
import numpy as np
from sapien.physx import PhysxArticulation, PhysxArticulationLinkComponent, PhysxArticulationJoint
from robotics.utils.sapien_utils import get_rigid_dynamic_component

from .gpu_engine import tensor2pose, pose2tensor
def to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
    

def pose2numpy(pose: Pose):
    return np.array([pose.p, pose.q])

class CPUEngine(Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = 'numpy'

    def sync_pose(self):
        pass

    def reset(self):
        pass
        

    def get_link_pose(self, link: PhysxArticulationLinkComponent):
        return pose2numpy(link.get_entity_pose())

    def get_pose(self, actor: Entity):
        return pose2numpy(actor.get_pose())

    def set_pose(self, actor: Entity, pose: Pose|np.ndarray| Tensor):
        _pose = tensor2pose(pose)
        actor.set_pose(_pose) # type: ignore

    def get_velocity(self, actor: Entity):
        body = get_rigid_dynamic_component(actor)
        assert body is not None, "Entity does not have a rigid dynamic component"
        return np.concatenate([body.get_linear_velocity(), body.get_angular_velocity()])

    def set_velocity(self, actor: Entity, velocity):
        body = get_rigid_dynamic_component(actor)
        assert body is not None, "Entity does not have a rigid dynamic component"
        body.set_linear_velocity(velocity[:3])
        body.set_angular_velocity(velocity[3:])


    def get_qpos(self, articulated: PhysxArticulation):
        return articulated.get_qpos()

    def set_qpos(self, articulated: PhysxArticulation, qpos):
        articulated.set_qpos(qpos)

    def set_qf(self, articulated: PhysxArticulation, qf):
        articulated.set_qf(qf)

    def get_qvel(self, articulated: PhysxArticulation):
        return articulated.get_qvel()

    def set_qvel(self, articulated: PhysxArticulation, qvel):
        articulated.set_qvel(qvel)

    def get_root_pose(self, articulated: PhysxArticulation):
        return articulated.get_root_pose()

    def set_root_pose(self, articulated: PhysxArticulation, pose):
        articulated.set_root_pose(tensor2pose(pose))

    def set_joint_target(self, articulated: PhysxArticulation, joints: list[PhysxArticulationJoint], joint_indices, targets):
        for joint, target in zip(joints, targets):
            joint.set_drive_target(target)

    def set_joint_velocity_target(self, articulated: PhysxArticulation, joints: list[PhysxArticulationJoint], joint_indices, targets):
        for joint, target in zip(joints, targets):
            joint.set_drive_velocity_target(target)

    def get_joint_target(self, articulated: PhysxArticulation):
        return [joint.get_drive_target() for joint in articulated.get_joints()]

    def init_batch(self, *args, **kwargs):
        raise NotImplementedError("Batch simulation is not supported in CPU engine")

    def set_cameras(self, scenes: list[sapien.Scene], cameras: list[list[sapien.render.RenderCameraComponent]]):
        self.scenes = scenes
        self.cameras = cameras

    def step_scenes(self, scenes: list[Scene]):
        scenes[0].step()

    def take_picture(self, scene_id: int = 0):
        self.scenes[scene_id].update_render()
        pictures = {}
        for i, cam in enumerate(self.cameras[scene_id]):
            #self.renderer.take_picture(cam, self.sub_scene[i])
            cam.take_picture()
            pictures[cam.name] = {
                'Color': cam.get_picture("Color"),
            }
        return pictures

    def clear(self):
        pass