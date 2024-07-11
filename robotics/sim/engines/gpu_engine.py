from __future__ import annotations

import torch
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
import numpy as np
from robotics.cfg import Config
import sapien
from sapien import physx, Pose
from sapien.physx import PhysxArticulation
from typing import cast


class GPUMemoryConfig(Config):
    """A gpu memory configuration dataclass that neatly holds all parameters that configure physx GPU memory for simulation"""

    temp_buffer_capacity: int = 2**25
    """Increase this if you get 'PxgPinnedHostLinearMemoryAllocator: overflowing initial allocation size, increase capacity to at least %.' """
    max_rigid_contact_count: int = 524288 * 40
    max_rigid_patch_count: int = (
        2**22
    )
    heap_capacity: int = 2**26
    found_lost_pairs_capacity: int = (
        2**25
    )  # 262144 is SAPIEN default but most tasks work with 2**25
    found_lost_aggregate_pairs_capacity: int = 2**20
    total_aggregate_pairs_capacity: int = 2**20


class GPUEngineConfig(Config):
    gpu_config: GPUMemoryConfig = GPUMemoryConfig()
    spacing: float = 30.


def tensor2pose(pose: np.ndarray | Pose | Tensor) -> Pose | list[Pose]:
    if isinstance(pose, Pose):
        return pose
    if isinstance(pose, Tensor):
        pose = cast(np.ndarray, pose.detach().cpu().numpy())

    if pose.ndim == 1:
        return Pose(pose[:3], pose[3:])
    else:
        return [Pose(p[:3], p[3:]) for p in pose]

    
def to_tensor(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=device)
    return x.to(device)


def pose2tensor(pose: Pose | np.ndarray | Tensor, device: torch.device) -> Tensor:
    if isinstance(pose, Pose):
        return to_tensor(np.concatenate([pose.p, pose.q]).astype(np.float32), device)
    return to_tensor(pose, device).float()

class RecordBase:
    attr_list: list[str]
    engine: "GPUEngine"

    def __init__(self, engine: "GPUEngine") -> None:
        self.engine = engine
        self._gpu_index = None
        self.clear()

    def clear(self):
        # the buffer is only used when the engine is not initialized
        self.buffer = {}

    def update(self):
        for k, (v, mask) in self.buffer.items():
            self._update(k, self.get_index(mask), to_tensor(v, self.engine.device).float())

    def set_value(self, name, value: torch.Tensor, mask = None):
        if not self.engine._initialized:
            self.engine.updated_records.add(self)
            self.buffer[name] = (value, mask)
        else:
            self._update(name, self.get_index(mask), value)
    
    def get_value(self, name) -> torch.Tensor:
        self.engine.readed_records.add(self)
        if name in self.buffer:
            raise ValueError("Can not read data from GPU before initialization.")
            return self.buffer[name][0]
        return self._get(name, self.get_index(None))

    def _get(self, name: str, index: torch.Tensor | int):
        raise NotImplementedError

    def _update(self, name: str, index: torch.Tensor | int, value: torch.Tensor):
        raise NotImplementedError

    def get_gpu_index(self):
        raise NotImplementedError

    def get_index(self, mask):
        if self._gpu_index is None:
            assert self.engine._initialized, "GPU engine has not been initialized, one can not read data from GPU before initialization."
            self._gpu_index = self.get_gpu_index()
        if mask is None:
            return self._gpu_index
        return self._gpu_index[to_tensor(mask, self.engine.device)]
         

def ensure_rigid_comp(actor: sapien.Entity) -> sapien.physx.PhysxRigidDynamicComponent:
    for c in actor.components:
        if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
            return c
    raise ValueError("Entity does not have a rigid dynamic component.. static entities are not supported.")

class EntityRecord(RecordBase):
    def __init__(self, actor: sapien.Entity | list[sapien.Entity], engine: "GPUEngine") -> None:
        self.actor = actor
        super().__init__(engine)

    def get_gpu_index(self):
        actor = self.actor
        if isinstance(actor, list):
            return to_tensor(np.array([ensure_rigid_comp(a).gpu_index for a in actor]), self.engine.device)
        else:
            self.comp = ensure_rigid_comp(actor)
            return self.comp.gpu_index

    def _get(self, name, index):
        if name == "pose":
            return  self.engine.rigid_dynamic_data[index, 0:7]
        elif name == 'velocity':
            return self.engine.rigid_dynamic_data[index, 7:]
        else:
            raise AttributeError(f"Attribute {name} not found")

    def _update(self, name: str, index: torch.Tensor | int, value: torch.Tensor):
        self.engine.updated_gpu_data_list.add("rigid_dynamic_data")
        if name == "pose":
            self.engine.rigid_dynamic_data[index, 0:7] = value
        elif name == 'velocity':
            self.engine.rigid_dynamic_data[index, 7:] = value
        else:
            raise AttributeError(f"Attribute {name} not found")


class ArticluatedRecord(RecordBase):
    def __init__(self, articulated: PhysxArticulation | list[PhysxArticulation], engine: "GPUEngine") -> None:
        articulated0 = articulated if isinstance(articulated, PhysxArticulation) else articulated[0]
        self.articulated = articulated
        self.dof = articulated0.dof
        super().__init__(engine)

    def get_gpu_index(self):
        if isinstance(self.articulated, list):
            return to_tensor(np.array([a.gpu_index for a in self.articulated]), self.engine.device)
        else:
            return self.articulated.gpu_index

    def _get(self, name: str, index: torch.Tensor|int):
        if name.startswith('q') or name.startswith('target'):
            return self.engine.articulation_data(name)[index, :self.dof]
        elif name == 'root_pose':
            return self.engine.articulation_data('link_pose')[index, 0, 0:7]
        else:
            raise AttributeError(f"Attribute {name} not found")
    
    def _update(self, name: str, index: torch.Tensor|int, value: torch.Tensor):
        self.engine.updated_gpu_data_list.add(f'articulation_{name}')
        if name == 'root_pose':
            self.engine.articulation_data('link_pose')[index, 0, 0:7] = value
            return
        data = self.engine.articulation_data(name)
        if name.startswith('q') or name.startswith('target'):
            data[index, :self.dof] = value
        else:
            raise AttributeError(f"Attribute {name} not found")


class GPUEngine:
    """_summary_
    For GPU engine:

    Can not read data from GPU before initialization, but we can write data and read it back.

    For GPU engine:
    1. Must reset the engine before the first step
    2. Must sync the pose before rendering
    3. Call before_step before the simulation step and after_step after the simulation step

    
    For batched envs (with scene number greater than 1):
    1. create multiple scenes
    2. set _scene of the simulator and load elements one by one (it's the users' duty to make the loader faster ..)  
    3. after loading, obtain the object lists by calling simulator.elements.get_actors() and simulator.elements.get_articulations()
    4. we create recoder for those object lists and from now on any operation on those objects through the engine will be broadcasted to all the scenes.
    """
    def __init__(self, config: GPUEngineConfig) -> None:
        if not physx.is_gpu_enabled():
            physx.enable_gpu()

        self.device = torch.device(
            "cuda"
        )  # TODO (stao): fix this for multi gpu support?
        gpu_config = config.gpu_config
        if isinstance(gpu_config, GPUMemoryConfig):
            gpu_config_dict = gpu_config.to_dict()
        else:
            assert isinstance(gpu_config, DictConfig)
            gpu_config_dict: dict = OmegaConf.to_container(gpu_config, resolve=True) # type: ignore
        physx.set_gpu_memory_config(**gpu_config_dict)
        self.px = physx.PhysxGpuSystem()

        self.scene_idx = 0
        self.config = config
        self._scene_config = None
        self.sub_scene: list[sapien.Scene] = []


        self.records: dict[sapien.Entity|PhysxArticulation, RecordBase] = {}
        self._initialized = False
        self.clear_updated()


    def clear_updated(self):
        self.updated_gpu_data_list = set()
        self.updated_records = set[RecordBase]()  # store things in the records but not sync with the gpu data.. only useful when the engine is not initialized 
        self.readed_records = set[RecordBase]() # store the value readed

    def clear(self):
        self.scene_idx = 0
        self.sub_scene = []
        self.clear_updated()
        self._initialized = False

        
    def get_record(self, actor: sapien.Entity | PhysxArticulation):
        if actor in self.records:
            return self.records[actor]
        else:
            if isinstance(actor, sapien.Entity):
                record = EntityRecord(actor, self)
            else:
                record = ArticluatedRecord(actor, self)
            self.records[actor] = record
            return record

    def reset(self):
        self._initialized = True
        self.px.gpu_init()

        
        def safe_torch(x) -> torch.Tensor:
            if x.shape[0] == 0:
                return cast(torch.Tensor, None)
            return x.torch()

        self.cuda_articulation_qpos = safe_torch(self.px.cuda_articulation_qpos)
        self.cuda_articulation_qvel = safe_torch(self.px.cuda_articulation_qvel)
        self.cuda_rigid_body_data = safe_torch(self.px.cuda_rigid_body_data)

        self.cuda_articulation_qf = safe_torch(self.px.cuda_articulation_qf)
        self.cuda_articulation_target_qpos = safe_torch(self.px.cuda_articulation_target_qpos)
        self.cuda_articulation_target_qvel = safe_torch(self.px.cuda_articulation_target_qvel)

        self.cuda_articulation_link_data = safe_torch(self.px.cuda_articulation_link_data)

        # clear robot qpos
        if self.cuda_articulation_qpos is not None:
            self.cuda_articulation_qpos[:] = 0.
            self.cuda_articulation_qvel[:] = 0.
            self.cuda_articulation_qf[:] = 0.

            self.px.gpu_apply_articulation_qf()
            self.px.gpu_apply_articulation_qpos()
            self.px.gpu_apply_articulation_qvel()

        if self.cuda_rigid_body_data is not None:
            # clear velocity
            self.cuda_rigid_body_data[..., 7:] = 0.
            self.px.gpu_apply_rigid_dynamic_data()


        #self.step_scenes(self.sub_scene) # step once to initialize the gpu data
        self.before_step()



    def sync_pose(self):
        self.px.sync_poses_gpu_to_cpu()

    def get_link_pose(self, link: physx.PhysxArticulationLinkComponent):
        self.px.gpu_update_articulation_kinematics()
        articulated = link.articulation
        record = self.get_record(articulated)
        self.px.gpu_fetch_articulation_link_pose()
        link_pose =  self.articulation_data('link_pose')[record.get_index(None), link.index]
        return link_pose[..., :7]

    def set_renderer(self, renderer):
        self.renderer = renderer

    def create_scene(self, scene_config: sapien.SceneConfig):
        if self._scene_config is None:
            self._scene_config = scene_config
            physx.set_scene_config(scene_config)

        scene = sapien.Scene(
            systems=[self.px, sapien.render.RenderSystem()]
        )
        self.px.set_scene_offset(
            scene, [self.scene_idx * self.config.spacing, 0, 0],
        )
        self.scene_idx += 1
        self.sub_scene.append(scene)
        return scene

    @property
    def rigid_dynamic_data(self):
        assert self._initialized, "GPU engine has not been initialized, one can not read data from GPU before initialization."
        if getattr(self, "_rigid_dynamic_data", None) is None:
            self.px.gpu_fetch_rigid_dynamic_data()
            self._rigid_dynamic_data = self.cuda_rigid_body_data
        return self._rigid_dynamic_data

    def articulation_data(self, name: str) -> torch.Tensor:
        assert self._initialized, "GPU engine has not been initialized, one can not read data from GPU before initialization."
        attr_name = f"_articulation_{name}"

        if name in ['target_position', 'target_velocity', 'qf']:
            return getattr(self, f"cuda_articulation_{name}")

        if getattr(self, attr_name, None) is None:
            getattr(self.px, f"gpu_fetch_articulation_{name}")()
            setattr(self, attr_name, getattr(self, f"cuda_articulation_{name.replace('link_pose', 'link_data')}"))
        return getattr(self, attr_name)

    def sync_articulation_data(self, name: str):
        getattr(self.px, f"gpu_apply_articulation_{name}")()

    def clear_gpu_buffer(self):
        setattr(self, "_rigid_dynamic_data", None)
        for name in ["qpos", "qvel", "link_pose"]:
            setattr(self, f"_articulation_{name}", None)
        
    def after_step(self):
        self.clear_gpu_buffer()
        self.clear_updated()

    def before_step(self):
        # sync all data that has been written
        for record in self.updated_records:
            record.update()
        for record in self.readed_records | self.updated_records:
            record.clear()
        for name in self.updated_gpu_data_list:
            getattr(self.px, f"gpu_apply_{name}")()

    def step_scenes(self, scenes: list[sapien.Scene]):
        self.before_step()
        if len(scenes) == 1:
            scenes[0].step()
        else:
            self.px.step()
        self.after_step()


    def init_batch(self, actors: list[list[sapien.Entity]], articulations: list[list[PhysxArticulation]]):
        def merge(scene_object_lists: list[list[sapien.Entity]] | list[list[PhysxArticulation]]):
            n = len(scene_object_lists[0])
            for scene_objects in scene_object_lists:
                assert len(scene_objects) == n, "All scenes must have the same number of objects"

            for obj_id in range(n):
                obj_batch = [scene_objects[obj_id] for scene_objects in scene_object_lists]
                if isinstance(obj_batch[0], sapien.Entity):
                    batch_record = EntityRecord(obj_batch, self) # type: ignore
                else:
                    batch_record = ArticluatedRecord(obj_batch, self) # type: ignore

                data = {}
                for idx, obj in enumerate(obj_batch):
                    record = self.records.get(obj, None)
                    if record is not None:
                        for k, (v, mask) in record.buffer.items():
                            assert mask is None, "Masked data for single data ..."
                            if k not in data:
                                data[k] = [], []
                            data[k][0].append(v)
                            data[k][1].append(idx)
                    self.records[obj] = batch_record

                for k, (v, index) in data.items():
                    mask = to_tensor(index, self.device)
                    batch_record.set_value(k, np.stack(v), mask=mask)

        merge(actors)
        merge(articulations)



            
    def set_pose(self, actor: sapien.Entity, pose: Pose|np.ndarray|Tensor):
        self.get_record(actor).set_value("pose", pose2tensor(pose, self.device))

    def get_pose(self, actor: sapien.Entity):
        return self.get_record(actor).get_value("pose")

    def set_velocity(self, actor: sapien.Entity, velocity: np.ndarray | Tensor):
        self.get_record(actor).set_value("velocity", to_tensor(velocity, self.device))

    def get_velocity(self, actor: sapien.Entity):
        return self.get_record(actor).get_value("velocity")

    def set_qpos(self, articulated: PhysxArticulation, qpos):
        self.get_record(articulated).set_value("qpos", qpos)

    def set_qvel(self, articulated: PhysxArticulation, qvel):
        self.get_record(articulated).set_value("qvel", qvel)

    def get_qpos(self, articulated: PhysxArticulation):
        return self.get_record(articulated).get_value("qpos")

    def get_qvel(self, articulated: PhysxArticulation):
        return self.get_record(articulated).get_value("qvel")

    def set_root_pose(self, articulated: PhysxArticulation, pose: Pose):
        return self.get_record(articulated).set_value("root_pose", pose2tensor(pose, self.device))

    def get_root_pose(self, articulated: PhysxArticulation):
        return tensor2pose(self.get_record(articulated).get_value("root_pose"))

    def set_qf(self, articulated: PhysxArticulation, qf: np.ndarray | Tensor):
        self.get_record(articulated).set_value("qf", to_tensor(qf, self.device))

    def set_joint_val(self, tensor: torch.Tensor, x:torch.Tensor|int, y: list[int], target: np.ndarray | Tensor):
        val = to_tensor(target, self.device).float()
        if isinstance(x, int):
            tensor[x, y] = val
        else:
            v = tensor[x]
            v[:, y] = val
            tensor[x] = v
    
    def set_joint_target(self, articulated: PhysxArticulation, joints, joint_indices, target: np.ndarray | Tensor):
        gpu_index = self.get_record(articulated).get_index(None)
        self.updated_gpu_data_list.add("articulation_target_position")
        self.set_joint_val(self.cuda_articulation_target_qpos, gpu_index, joint_indices, target)

    def set_joint_velocity_target(self, articulated: PhysxArticulation, joints, joint_indices, target: np.ndarray | Tensor):
        gpu_index = self.get_record(articulated).get_index(None)
        self.updated_gpu_data_list.add("articulation_target_velocity")
        self.set_joint_val(self.cuda_articulation_target_qvel, gpu_index, joint_indices, target)
    
    def get_joint_target(self, articulated: PhysxArticulation):
        gpu_index = self.get_record(articulated).get_index(None)
        return self.cuda_articulation_target_qpos[gpu_index]


    # ----------------- Rendering -----------------
    def set_cameras(self, scenes: list[sapien.Scene], cameras: list[list[sapien.render.RenderCameraComponent]]):
        # TODO: the following code is not working, need to fix it for batched rendering
        # objects: list[physx.PhysxRigidBodyComponent | physx.PhysxArticulationLinkComponent] = []
        # for k in self.records:
        #     if isinstance(k, sapien.Entity):
        #         objects.append(ensure_rigid_comp(k))
        #     else:
        #         objects.extend(k.links)

        # for body in objects:
        #     rb = body.entity.find_component_by_type(sapien.render.RenderBodyComponent)
        #     if rb is None:
        #         continue
        #     rb = cast(sapien.render.RenderBodyComponent, rb)
        #     for s in rb.render_shapes:
        #         s.set_gpu_pose_batch_index(body.gpu_pose_index) # type: ignore

        # for cam in cameras:
        #     body = cam.entity.find_component_by_type(
        #         sapien.physx.PhysxRigidBodyComponent
        #     )
        #     if body is None:
        #         continue
        #     body = cast(physx.PhysxRigidBodyComponent, body)
        #     cam.set_gpu_pose_batch_index(body.gpu_pose_index) # type: ignore

        # # render system group manages batched rendering
        # self.render_system_group = sapien.render.RenderSystemGroup(
        #     [s.render_system for s in self.sub_scene]
        # )
        # # camera group renders images in batches
        # self.camera_group = self.render_system_group.create_camera_group(
        #     cameras, ["Color", "PositionSegmentation"]
        # )
        # self.render_system_group.set_cuda_poses(self.px.cuda_rigid_body_data)
        self.scenes = scenes
        self.cameras = cameras

    def take_picture(self, scene_id: int = 0):
        # NOTE: temporarily not supporting batched rendering.. and it seems very slow ..
        self.before_step()
        self.sync_pose()
        self.scenes[scene_id].update_render()
        pictures = {}
        for i, cam in enumerate(self.cameras[scene_id]):
            #self.renderer.take_picture(cam, self.sub_scene[i])
            cam.take_picture()
            pictures[cam.name] = {
                'Color': cam.get_picture("Color"),
            }
        return pictures