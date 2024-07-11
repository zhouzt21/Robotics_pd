import numpy as np
from pathlib import Path
from ..entity import Entity
from sapien.render import RenderCameraComponent
from .controller import Controller, ControllerConfig
from functools import cached_property
import sapien
from sapien.physx import PhysxArticulation
from ..sensors.sensor_base import SensorBase
from sapien import physx
from robotics.utils.sapien_utils import check_urdf_config, parse_urdf_config, apply_urdf_config
from typing import List, Union, Sequence, Dict, Tuple, TYPE_CHECKING
import gymnasium as gym
import sapien.wrapper.urdf_loader
from functools import cached_property


if TYPE_CHECKING:
    from ..simulator import Simulator


class Robot(Entity):
    arm: Controller
    base: Controller
    gripper: Controller
    ignore_srdf: bool = False

    def __init__(self,
        control_freq: int,
        urdf_path: str,
        urdf_config: dict,
        balance_passive_force: bool = True,
        fix_root_link=True,
    ) -> None:
        object.__setattr__(self, '__children__', {})

        self.control_freq = control_freq
        self.fix_root_link = fix_root_link
        self.balance_passive_force = balance_passive_force

        self.urdf_path = urdf_path
        self.urdf_config = urdf_config

        self.controllers: Dict[str, Controller] = {}
        self.sensors = {}


    def generate_srdf(self, collision_groups, robot_name):
        # <disable_collisions link1="joint1" link2="rplidar_link" reason="Default"/>
        TEMPLATE = """<?xml version="1.0" ?>
<robot name="ROBOTNAME">
LIST
</robot>"""

        line = '<disable_collisions link1="A" link2="B" reason="Default"/>'
        TEMPLATE = TEMPLATE.replace('ROBOTNAME', robot_name)
        lines = []
        for group in collision_groups:
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    lines.append(line.replace('A', group[i]).replace('B', group[j]))
        TEMPLATE = TEMPLATE.replace('LIST', '\n'.join(lines))
        return TEMPLATE


    def get_urdf_path(self):
        return self.urdf_path

    def get_srdf_path(self):
        return self.urdf_path[:-5] + '.srdf'


    def _load_by_builders(self, articulation_builders, actor_builders, cameras,
    ) -> PhysxArticulation:
        # articulation_builders, actor_builders, cameras = self.parse(
        #     urdf_file, srdf_file, package_dir
        # )
        if len(articulation_builders) > 1 or len(actor_builders) != 0:
            raise Exception(
                "URDF contains multiple objects, call load_multiple instead"
            )

        articulations = []
        for b in articulation_builders:
            articulations.append(b.build())

        actors = []
        for b in actor_builders:
            actors.append(b.build())

        name2entity = dict()
        for a in articulations:
            for l in a.links:
                name2entity[l.name] = l.entity

        for a in actors:
            name2entity[a.name] = a

        assert len(cameras) == 0, "Camera is not supported yet"
        for cam in cameras:
            cam_component = RenderCameraComponent(cam["width"], cam["height"])
            if cam["fovx"] is not None and cam["fovy"] is not None:
                cam_component.set_fovx(cam["fovx"], False)
                cam_component.set_fovy(cam["fovy"], False)
            elif cam["fovy"] is None:
                cam_component.set_fovx(cam["fovx"], True)
            elif cam["fovx"] is None:
                cam_component.set_fovy(cam["fovy"], True)

            cam_component.near = cam["near"]
            cam_component.far = cam["far"]
            cam_component.local_pose = cam["pose"]
            name2entity[cam["parent"]].add_component(cam_component)

        return articulations[0]        
    
    def _load(self, world: "Simulator"):
        self.engine = world._engine
        if not hasattr(self, 'loader'):
            scene = world._scene

            urdf_path = str(self.urdf_path)

            urdf_config = parse_urdf_config(self.urdf_config, scene)
            check_urdf_config(urdf_config)

            # TODO(jigu): support loading multiple convex collision shapes


            self.srdf = None if not self.ignore_srdf else ""

            self.urdf_path = urdf_path

            loader = scene.create_urdf_loader()
            loader.fix_root_link = self.fix_root_link
            apply_urdf_config(loader, urdf_config)
            self._update_loader(loader)

            self.loader = loader
            self.builders = loader.parse(self.urdf_path, self.srdf)
        else:
            loader = self.loader
            articulation_builders, actor_builders, cameras = self.builders
            for b in articulation_builders:
                b.set_scene(world._scene)
            for b in actor_builders:
                b.set_scene(world._scene)
            self.loader.set_scene(world._scene)


        self.articulation: physx.PhysxArticulation = self._load_by_builders(*self.builders)
        assert self.articulation is not None, f"Fail to load URDF from {urdf_path}"
        self.articulation.set_name(Path(self.urdf_path).stem)

        # Cache robot link ids
        if not hasattr(self, 'robot_link_ids'):
            self.robot_link_ids = [link.name for link in self.articulation.get_links()]

        self._post_load(world)

    def _update_loader(self, loader: sapien.wrapper.urdf_loader.URDFLoader):
        pass

    def _post_load(self, world: "Simulator"):
        pass

        
    def _get_sapien_entity(self) -> List[Union[Entity, physx.PhysxArticulation]]:
        return [self.articulation]

    def get_active_joints(self, joint_names: Sequence[Union[str, int]]) -> Tuple[List[int], List[physx.PhysxArticulationJoint]]:
        articulation = self.articulation
        joints = articulation.get_active_joints()
        _joint_names = [x.name for x in joints]

        joint_indices = []
        for x in joint_names:
            if isinstance(x, str):
                assert x in _joint_names, f"Cannot find joint {x} in {self} with joints {_joint_names}"
                joint_indices.append(_joint_names.index(x))
            else:
                joint_indices.append(x)
        return joint_indices, [joints[idx] for idx in joint_indices]


    def _before_simulation_step(self, *args, **kwargs):
        if self.balance_passive_force:
            qf = self.articulation.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        else:
            qf = np.zeros(self.articulation.dof)
        self.engine.set_qf(self.articulation, qf)
        

    def get_state(self):
        qpos = self.engine.get_qpos(self.articulation)
        qvel = self.engine.get_qvel(self.articulation)
        controllers = {}
        for k, v in self.controllers.items():
            if (state := v.get_state()) is not None:
                controllers[k] = state
        output: dict =  {
            'qpos': qpos, 'qvel': qvel, 
        }
        if len(controllers) > 0:
            output['controllers'] = controllers
        return output

    def set_state(self, state):
        self.engine.set_qpos(self.articulation, state['qpos'])
        self.engine.set_qvel(self.articulation, state['qvel'])
        if 'controllers' in state:
            for k, v in state['controllers'].items():
                self.controllers[k].set_state(v)


    def register_controller(self, name: str, controller: Controller):
        assert name in ["arm", "gripper", "base"]
        self.controllers[name] = controller
        self.add_subentity(name, controller)
        setattr(self, name, controller)

    def _find(self, uid: str):
        assert '/' not in uid
        links = self.articulation.get_links()
        all_names = []
        for i in links:
            all_names.append(i.name)
            if i.name == uid:
                return i
        joints = self.articulation.get_joints()
        for i in joints:
            all_names.append(i.name)
            if i.name == uid:
                return i
        raise FileNotFoundError(f'Cannot find {uid} in the agent {self} with names {all_names}')

    @cached_property
    def action_space(self):
        """Flat multiple Box action spaces into a single Box space."""
        action_spaces = {k: v.action_space for k, v in self.controllers.items()}

        action_dims = []
        low = []
        high = []
        action_mapping = dict()
        offset = 0

        for action_name, action_space in action_spaces.items():
            if isinstance(action_space, gym.spaces.Box):
                assert len(action_space.shape) == 1, (action_name, action_space)
            else:
                raise TypeError(action_space)

            action_dim = action_space.shape[0]
            action_dims.append(action_dim)
            low.append(action_space.low)
            high.append(action_space.high)
            action_mapping[action_name] = (offset, offset + action_dim)
            offset += action_dim

        _action_space = gym.spaces.Box(
            low=np.hstack(low),
            high=np.hstack(high),
            shape=[sum(action_dims)],
            dtype=np.float32,
        )
        self._action_mapping = action_mapping
        return _action_space

        
    def set_action(self, action):
        if action is None:
            return
        self.action_space # make sure action space is initialized

        action_dict = {}
        for uid, controller in self.controllers.items():
            start, end = self._action_mapping[uid]
            controller.set_action(action[..., start:end])
        return action_dict

    def get_sensors(self):
        """
        sensors to be added to the simulator
        """
        return {}

    def get_ros_plugins(self):
        """
        ros plugins to be added to the simulator
        """
        return []

    @property
    def ee_name(self):
        raise NotImplementedError(f"Cannot get ee name of {type(self)}")

    @property
    def frame_mapping(self):
        raise NotImplementedError(f"Cannot get frame mapping of {type(self)}")

    @cached_property
    def pmodel(self):
        return self.articulation.create_pinocchio_model() # type: ignore

        
class MobileRobot(Robot):
    def __init__(self, control_freq: int, urdf_path: str, urdf_config: dict, balance_passive_force: bool = True, fix_root_link=True, motion_model='holonomic') -> None:
        self.motion_model = motion_model
        super().__init__(control_freq, urdf_path, urdf_config, balance_passive_force, fix_root_link)

    def set_base_pose(self, xy, ori):
        #qpos = self.articulation.get_qpos()
        qpos = self.engine.get_qpos(self.articulation)
        qpos[..., 0:2] = xy
        qpos[..., 2] = ori
        self.engine.set_qpos(self.articulation, qpos)

    def get_base_pose(self):
        qpos = self.engine.get_qpos(self.articulation)
        return qpos[..., 0:2], qpos[..., 2]

        
    @property
    def base_name(self):
        raise NotImplementedError(f"Cannot get base name of {type(self)}")

    def get_lidar_mount_frame(self):
        #raise NotImplementedError(f"Cannot get lidar frame of {type(self)}")
        return 'robot/' + self.base_name

    def build_lidar(self, **frame_kwargs):
        from ..sensors.lidar_v2 import Lidar, LidarConfig
        return Lidar(LidarConfig(
            **frame_kwargs,
            width=64,
            height=64,
            fov=np.pi / 2,
        ))

    def get_sensors(self) -> Dict[str, SensorBase]:
        # TODO: allow subclasses to override this
        raise NotImplementedError(f"Cannot get sensors of {type(self)}")