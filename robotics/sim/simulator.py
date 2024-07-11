from __future__ import annotations
"""
Simulator built based on SAPIEN3.
"""
from typing import cast, Union, TYPE_CHECKING, Optional

import logging
import numpy as np
import sapien.core as sapien

import torch
from .entity import Entity, Composite
from .simulator_base import SimulatorBase, FrameLike, SimulatorConfig


if TYPE_CHECKING:
    from .robot.robot_base import Robot
    from .ros_plugins.module import ROSModule


class Simulator(SimulatorBase):
    _loaded_eneity: set["Entity"]
    robot: Optional["Robot"]
    elements: "Composite"

    def __init__(
        self,
        config: SimulatorConfig,
        robot: Optional["Robot"],
        elements: dict[str, Union[Entity, dict]],
        add_ground: bool = True,
        ros_module: Optional["ROSModule"] = None,
    ):
        self.robot = robot
        self._setup_elements(elements)
        self.add_ground = add_ground and self._if_add_ground(elements)

        super().__init__(config)

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
    def control_freq(self):
        return self._control_freq

    @property
    def control_timestep(self):
        return 1.0 / self._control_freq

    @property
    def dt(self):
        return self.control_timestep

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