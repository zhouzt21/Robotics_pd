"""
Simulator built based on SAPIEN3. Already add cloth sim now.
"""
import tqdm
from robotics.cfg import Config

import sapien.core as sapien
from typing import Optional, Union, Callable, Tuple
from robotics import Pose
from sapien.utils.viewer import Viewer
from sapien import physx

from .sensors.sensor_cfg import CameraConfig
from .engines.cpu_engine import CPUEngine
from .engines.gpu_engine import GPUEngineConfig, GPUEngine

from __future__ import annotations
from typing import cast, Union, TYPE_CHECKING, Optional

import logging
import numpy as np
import sapien.core as sapien

import torch
from .entity import Entity, Composite

# maiskill env
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs.types import SimConfig

from dataclasses import dataclass
import dacite


if TYPE_CHECKING:
    from .robot.robot_base import Robot
    from .ros_plugins.module import ROSModule

_Engine: CPUEngine | GPUEngine | None = None
def get_engine() -> CPUEngine | GPUEngine:
    assert _Engine is not None, "Engine is not initialized. Please create a simulator first."
    return _Engine

 #replace the simulator, other module (like ros) recognize as world 
 # "sim" - origin simulator code, "clo" - clothenv code, "both" - for both
 
 #for coder to change sim config easily
class PDSimConfig(Config):   # config for simulator 
    # sim_freq: int = 500    #TODO add into sys config
    shader_dir: str = "default" 
    enable_shadow: bool = False
    contact_offset: float = 0.02

    control_freq_without_robot: int = 20

    # viewer config 
    viewer_camera: Optional[CameraConfig] = None
    viewer_id: Optional[int] = None

    # engine config
    gpu_config: Optional[GPUEngineConfig] = None

    # solver config
    solver_iterations: int = 50
    velocity_iterations: int = 1
    enable_pcm: bool = False

    n_scenes: int = 1


class PDSimulator(BaseEnv):
    # sim 
    # robot: Optional["Robot"]
    env_cfg: "PDSimConfig"  # set the true config when init（）[why using ""?]
    _loaded_eneity: set["Entity"]
    elements: "Composite"

    # clo
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    # both
    def __init__(self, 
                *args,
                robot_uids ="panda",
                sim_cfg: Union[SimConfig, dict] = dict(),
                env_cfg : PDSimConfig | dict = PDSimConfig(),
                elements: dict[str, Union[Entity, dict]],
                add_ground: bool = True,
                ros_module: Optional["ROSModule"] = None,
                **kwargs,  
    ):
        # clo
        if isinstance(env_cfg, PDSimConfig):
            self._env_cfg = env_cfg
        else:
            self._env_cfg = dacite.from_dict(data_class=PDSimConfig, data=env_cfg, config=dacite.Config(strict=True))
        
        self.shader_dir = env_cfg.shader_dir
        self.enable_shadow = env_cfg.enable_shadow
        self.contact_offset = env_cfg.contact_offset

        # sim  
        renderer_kwargs = {}
        self._renderer = sapien.SapienRenderer(**renderer_kwargs)
        self._viewer: Optional[Viewer] = None  
        self._viewer_camera = env_cfg.viewer_camera
        self._viewer_id = env_cfg.viewer_id

        # clo
        self._engine = CPUEngine() if env_cfg.gpu_config is None else GPUEngine(env_cfg.gpu_config)
        global _Engine
        _Engine = self._engine

        self._engine.set_renderer(self._renderer)  # sim

        # simor
        # self.robot = robot
        self._setup_elements(elements)
        self.add_ground = add_ground and self._if_add_ground(elements)

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        control_freq = self._control_freq = robot.control_freq if robot is not None else config.control_freq_without_robot
        if self._sim_freq % control_freq != 0:
            logging.warning(
                f"sim_freq({self._sim_freq}) is not divisible by control_freq({control_freq}).",
            )
        self._sim_steps_per_control = self._sim_freq // control_freq

        self.modules: list['ROSModule'] = []
        if ros_module is not None:
            self.modules.append(ros_module)
        for m in self.modules:
            m.set_sim(self)

    @property
    def sim_freq(self):
        return self._sim_freq

    @property
    def sim_timestep(self):
        return 1.0 / self._sim_freq

    @property
    def control_freq(self):
        return self._control_freq

    @property
    def control_timestep(self):
        return 1.0 / self._control_freq

    @property
    def dt(self):
        return self.control_timestep
    
    # -------------------------------------------------------------------------- #
    # Code for set scene and reset() function.
    # -------------------------------------------------------------------------- #
   
    def set_scene(self, idx: int):   #TODO
        assert 0 <= idx < self.config.n_scenes
        self._scene = self._scene_list[idx]
        self._scene_idx = idx
        self._viewer_has_scene_updated = False
        if self._viewer is not None:
            self._viewer.set_scene(self._scene)
    
    def set_viewer_scenes(self, idx: list[int], spacing: float = 1.): #TODO
        if spacing is not None and len(idx) > 1:
            import numpy as np
            side = int(np.ceil(len(idx) ** 0.5))
            idx = np.arange(len(idx))
            offsets = np.stack([idx // side, idx % side, np.zeros_like(idx)], axis=1) * spacing
        else:
            offsets = None

        self.viewer.set_scenes([self._scene_list[i] for i in idx], offsets=offsets)
        vs = self.viewer.window._internal_scene # type: ignore
        cubemap = self._scene.render_system.get_cubemap()
        if cubemap is not None:
            vs.set_cubemap(cubemap._internal_cubemap)
        else:
            vs.set_ambient_light([0.5, 0.5, 0.5])
        self._setup_viewer(False)
        self._viewer_has_scene_updated = False

    def reset(self, init_engine: bool = True): #TODO
        """
        Although this is called reset, it is actually initilize the simulator and all scenes
        """
        self.close()

        self._setup_scene()

        self.actor_batches = []
        self.articulation_batches = []
        self.camera_batches = []

        if self.config.n_scenes > 1:
            it = tqdm.tqdm(self._scene_list, desc="Creating all scenes")
        else:
            it = self._scene_list

        for idx, scene in enumerate(it):
            self.set_scene(idx)
            self._setup_lighting()
            # self._scene.load_widget_from_package("demo_arena", "DemoArena")
            self._load()

        self._scene = self._scene_list[0]

        if init_engine:
            self._engine.reset()
        if len(self._scene_list) > 1:
            self._engine.init_batch(self.actor_batches, self.articulation_batches)
        
        if len(self.camera_batches) > 0:
            self._engine.set_cameras(self._scene_list, self.camera_batches)

    def _load(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Code for setup sapien scene and renderer
    # -------------------------------------------------------------------------- #
    def _get_default_scene_config(self):
        scene_config = sapien.SceneConfig()
        # note these frictions are same as unity
        physx.set_default_material(dynamic_friction=0.5, static_friction=0.5, restitution=0)
        scene_config.contact_offset = self.contact_offset
        scene_config.enable_pcm = self.config.enable_pcm
        scene_config.solver_iterations = self.config.solver_iterations
        # NOTE(fanbo): solver_velocity_iterations=0 is undefined in PhysX
        scene_config.solver_velocity_iterations = self.config.velocity_iterations
        return scene_config

    def _setup_scene(self, scene_config: Optional[sapien.SceneConfig] = None):
        """Setup the simulation scene instance.
        The function should be called in reset(). Called by `self.reconfigure`"""
        if scene_config is None:
            scene_config = self._get_default_scene_config()

        def create_scene():
            scene = self._engine.create_scene(scene_config)
            scene.set_timestep(1.0 / self._sim_freq)
            return scene

        self._scene = create_scene()

        self._scene_list = [self._scene]
        if self.config.n_scenes > 1:
            it = range(1, self.config.n_scenes)
            self._scene_list += [create_scene() for i in it]
        self._viewer_has_scene_updated = False


    def close(self):
        """Clear the simulation scene instance and other buffers.
        The function can be called in reset() before a new scene is created. 
        Called by `self.reconfigure` and when the environment is closed/deleted
        """
        self._close_viewer()
        setattr(self, "_scene", None)
        setattr(self, "_scene_list", None)

    def _close_viewer(self):
        if self._viewer is None:
            return
        self._viewer.close()
        self._viewer = None


    def _add_ground(self, altitude=0.0, render=True):
        if render:
            rend_mtl = self._renderer.create_material()
            rend_mtl.base_color = [0.06, 0.08, 0.12, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
        else:
            rend_mtl = None
        ground = self._scene.add_ground(
            altitude=altitude,
            render=render,
            render_material=rend_mtl,
        )

        self.ground = ground
        return ground


    @property
    def viewer(self):
        if self._viewer is None:
            self._viewer = Viewer(self._renderer)
            self._setup_viewer()
        return self._viewer


    def _setup_lighting(self):
        """Setup lighting in the scene. Called by `self.reconfigure`"""

        shadow = self.enable_shadow
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._light1 = self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )
        self._light2 = self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _set_viewer_camera(self, camera_config: Optional["CameraConfig"]):
        from .sensors.camera_v2 import CameraConfig
        from .sensors.sensor_base import get_pose_from_sensor_cfg

        if camera_config is None:
            camera_config = CameraConfig(p=(1, 1, 1), look_at=(0, 0, 0.5))
        import transforms3d
        pose = get_pose_from_sensor_cfg(camera_config)
        xyz, q = pose.p, pose.q
        rpy = transforms3d.euler.quat2euler(q)
        self._viewer.set_camera_xyz(*xyz) # type: ignore
        self._viewer.set_camera_rpy(rpy[0], -rpy[1], -rpy[2]) # type: ignore  mysterious

    def _setup_viewer(self, set_scene: bool=True):
        #TOOD: setup viewer
        assert self._viewer is not None
        if set_scene:
            self._viewer.set_scene(self._scene)
        camera_config = self._viewer_camera
        self._set_viewer_camera(camera_config)
        self._viewer.control_window._show_camera_linesets = self._show_camera_linesets # type: ignore

        if self._viewer_id is not None:
            assert self._viewer.plugins is not None
            for i in self._viewer.plugins:
                if hasattr(i, 'camera_index'): # HACK: set camera index, maybe there is a better way
                    i.camera_index = self._viewer_id

    def update_scene_if_needed(self):
        if not self._viewer_has_scene_updated:
            self._engine.sync_pose()
            try:
                self.viewer.window.update_render() # type: ignore
            except AttributeError:
                self._scene.update_render() # in case of old sapien version
            self._viewer_has_scene_updated = True

    def render(self, show=True):
        self.update_scene_if_needed()
        if show:
            self.viewer.render()
        return self._viewer



    # ---------------------------------------------------------------------------- #
    # Setup scene with other entities and modules. Code for step()
    # ---------------------------------------------------------------------------- #

    def _if_add_ground(self, elements: dict[str, Union[Entity, dict]]):
        for v in elements.values():
            if hasattr(v, 'has_ground') and getattr(v, 'has_ground'):
                return False
        return True

    def _setup_elements(self, elements: dict[str, Union[Entity, dict]]):
        robot = self.robot
        if robot is not None:
            self.robot_cameras = robot.get_sensors()
            elements = dict(robot=robot, **self.robot_cameras, **elements)
        self.elements = Composite('', **elements)


    def find(self, uid=''):
        return self.elements.find(uid)

    def is_loaded(self, entity: "Entity"):
        return entity in self._loaded_eneity

    def load_entity(self, entity: "Entity"):
        if self.is_loaded(entity):
            return
        entity._load(self)
        self._loaded_eneity.add(entity)


    # load ground ..
    def _load(self):
        if self.add_ground:
            self._add_ground(render=True)
        self._loaded_eneity = set()
        self._elem_cache = {}
        self.elements.load(self)

        # TODO: maybe we can just use self._scene.get_actors() to get all the actors. However, I don't know if the order will be the same.
        self.actor_batches.append(self.elements.get_actors())
        self.articulation_batches.append(self.elements.get_articulations())
        self.camera_batches.append(self.elements.get_sapien_obj_type(sapien.render.RenderCameraComponent))

        for m in self.modules:
            m.load()


    def step(self, action: Union[None, np.ndarray, dict], print_contacts_for_debug: bool=False):
        self._viewer_has_scene_updated = False
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            assert self.robot is not None
            self.robot.set_action(action)
        else:
            raise TypeError(type(action))

        #TODO: ROS_MODULE.before_control_step 
        for m in self.modules:
            m.before_control_step()

        for _ in range(self._sim_steps_per_control):
            if self.robot is not None:
                self.robot.before_simulation_step()
            self._engine.step_scenes(self._scene_list)
            if print_contacts_for_debug:
                print(self._scene.get_contacts())

        for m in self.modules:
            m.after_control_step()
    

    # ---------------------------------------------------------------------------- #
    # Advanced: utilities for ROS2 and motion planning. I am not sure if we should 
    # put them here.
    # ---------------------------------------------------------------------------- #

    def gen_scene_pcd(self, num_points: int = int(1e5), exclude=()):
        """Generate scene point cloud for motion planning, excluding the robot"""
        pcds = []
        sim = self
        for k, v in sim.elements.items():
            if k != 'robot':
                out = v.get_pcds(num_points, exclude)
                if out is not None:
                    pcds.append(out)
        return np.concatenate(pcds)