"""
PDSimulator built based on SAPIEN3. Add robot agent and basic environment settings.
"""
from __future__ import annotations
from typing import cast, Union, TYPE_CHECKING, Optional

import logging
import numpy as np
import sapien.core as sapien

import os
import sys
sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from robotics.sim.entity import Entity, Composite

# maiskill env
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

from dataclasses import dataclass
import dacite


import tqdm
from robotics.cfg import Config

import sapien.core as sapien
from typing import Optional, Union, Callable, Tuple
from sapien.utils.viewer import Viewer
from sapien import physx

from robotics.sim.sensors.sensor_cfg import CameraConfig

if TYPE_CHECKING:
    from .robot.robot_base import Robot
    from .ros_plugins.module import ROSModule
 
class PDSimConfig(Config):   # config for simulator 
    sim_freq: int = 500    
    shader_dir: str = "default" 
    enable_shadow: bool = False
    contact_offset: float = 0.02

    control_freq_without_robot: int = 20

    # viewer config 
    viewer_camera: Optional[CameraConfig] = None
    viewer_id: Optional[int] = None

    # base env config 
    render_mode: str = None

    # solver config (not use temporarily)
    solver_iterations: int = 50
    velocity_iterations: int = 1
    enable_pcm: bool = False

    # entity config 
    add_ground: bool = True           
    
    # agent config
    robot_uids: str ="panda"
    control_mode: str = None
    ros_module: Optional["ROSModule"] = None

    interaction_links=("panda_rightfinger", "panda_leftfinger","panda_link7",
                        "panda_link6","panda_link5","panda_link4","panda_link3",
                        "panda_link2","panda_link1","panda_link0")
    robot_init_qpos_noise: int =0   #TODO add init_pose config


class PDSimulator(BaseEnv):
    
    env_cfg: "PDSimConfig" 
    # not use temporarily
    _loaded_eneity: set["Entity"]
    elements: "Composite"
    _show_camera_linesets = True
    
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda
    
    def __init__(self, env_cfg : PDSimConfig | dict = PDSimConfig(), elements: dict[str, Union[Entity, dict]] = {}   ):
         
        if isinstance(env_cfg, PDSimConfig):
            self._env_cfg = env_cfg
        else:
            self._env_cfg = dacite.from_dict(data_class=PDSimConfig, data=env_cfg, config=dacite.Config(strict=True))
        
        self._sim_freq = env_cfg.sim_freq   #  connect (sim_freq in sim_cfg is 100 default)        
        self.shader_dir = env_cfg.shader_dir
        self.enable_shadow = env_cfg.enable_shadow
        self.contact_offset = env_cfg.contact_offset
        self.control_freq_without_robot = env_cfg.control_freq_without_robot
        
        renderer_kwargs = {}
        self._renderer = sapien.SapienRenderer(**renderer_kwargs)
        self._viewer: Optional[Viewer] = None        
        self._viewer_camera = env_cfg.viewer_camera
        self._viewer_id = env_cfg.viewer_id

        self._setup_elements(elements)
        self.add_ground = env_cfg.add_ground and self._if_add_ground(elements)   

        self.modules: list['ROSModule'] = []     
        if env_cfg.ros_module is not None:
            self.modules.append(env_cfg.ros_module)   

        self.interaction_links = set(env_cfg.interaction_links)
        self.robot_init_qpos_noise= env_cfg.robot_init_qpos_noise

        super().__init__(robot_uids=env_cfg.robot_uids,render_mode=env_cfg.render_mode,control_mode=env_cfg.control_mode)           
        
        control_freq = self._control_freq = self.agent._control_freq if self.agent is not None else self.control_freq_without_robot
        if self._sim_freq % control_freq != 0:
            logging.warning(
                f"sim_freq({self._sim_freq}) is not divisible by control_freq({control_freq}).",
            )
        self._sim_steps_per_control = self._sim_freq // control_freq

        # add for ros compatibility (haven't check yet) TODO
        self.robot = self.agent 

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

    # ---------------------------------------------------------------------------- #
    # Setup scene with viewer and renderer. already finished check
    # ---------------------------------------------------------------------------- #

    @property
    def viewer(self):
        if self._viewer is None:
            self._viewer = Viewer(self._renderer)
            self._setup_viewer()
        return self._viewer

    def _set_viewer_camera(self, camera_config: Optional["CameraConfig"]):
        from robotics.sim.sensors.camera_v2 import CameraConfig
        from robotics.sim.sensors.sensor_base import get_pose_from_sensor_cfg

        if camera_config is None:
            camera_config = CameraConfig(p=(1, 1, 1), look_at=(0, 0, 0.5))
        import transforms3d
        pose = get_pose_from_sensor_cfg(camera_config)
        xyz, q = pose.p, pose.q
        rpy = transforms3d.euler.quat2euler(q)
        self._viewer.set_camera_xyz(*xyz) # type: ignore
        self._viewer.set_camera_rpy(rpy[0], -rpy[1], -rpy[2]) # type: ignore  mysterious

    def _setup_viewer(self, set_scene: bool=True):
        # setup viewer, actually called by ''_reconfigure()''
        assert self._viewer is not None
        if set_scene:
            for s in self._scene.sub_scenes:
                self._viewer.set_scene(s)
            # self._viewer.set_scene(self._scene)
        camera_config = self._viewer_camera
        self._set_viewer_camera(camera_config)
        self._viewer.control_window._show_camera_linesets = self._show_camera_linesets # type: ignore

        if self._viewer_id is not None:
            assert self._viewer.plugins is not None
            for i in self._viewer.plugins:
                if hasattr(i, 'camera_index'): # HACK: set camera index, maybe there is a better way
                    i.camera_index = self._viewer_id

    # ---------------------------------------------------------------------------- #
    # Setup scene with other entities(basic environ setting) and modules loaded. 
    # ---------------------------------------------------------------------------- #

    def _add_ground(self, altitude=0.0, render=True):
        if render:
            rend_mtl = self._renderer.create_material()
            rend_mtl.base_color = [0.06, 0.08, 0.12, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
        else:
            rend_mtl = None
        ground = self._scene.sub_scenes[0].add_ground(  # change into the first sub_scene
            altitude=altitude,
            render=render,
            render_material=rend_mtl,
        )

        self.ground = ground
        return ground
    
    def _if_add_ground(self, elements: dict[str, Union[Entity, dict]]):
        for v in elements.values():
            if hasattr(v, 'has_ground') and getattr(v, 'has_ground'):
                return False
        return True

    # uncompatible with _load_scene
    # TODO setup agent robot's sensors(change to agent's sensors)
    def _setup_elements(self, elements: dict[str, Union[Entity, dict]]):
        # robot = self.agent
        # if robot is not None:
        #     self.robot_cameras = robot.get_sensors()
        #     elements = dict(robot=robot, **self.robot_cameras, **elements)
        self.elements = Composite('', **elements)


    def find(self, uid=''):
        return self.elements.find(uid)

    # not use temporarily
    def is_loaded(self, entity: "Entity"):
        return entity in self._loaded_eneity

    # not use temporarily
    def load_entity(self, entity: "Entity"):
        if self.is_loaded(entity):
            return
        entity._load(self)
        self._loaded_eneity.add(entity)

    # load ground .. (change _load() into _load_scene)  also can add other basic environ setting 
    # also load ros module (other entities are loaded in pdcloth_env.py)
    def _load_scene(self):
        if self.add_ground:
            self._add_ground(render=True)
        
        # not use temporarily
        self._loaded_eneity = set()
        self._elem_cache = {}
        self.elements.load(self)

        for m in self.modules:
            m.load()

    # TODO:use origin step() but add ros module control
    def _before_control_step(self):
        for m in self.modules:
            m.before_control_step()
        
    def _after_control_step(self):
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