from typing import Dict, TYPE_CHECKING, Optional, Sequence
import sapien.core as sapien
import numpy as np
from ..environ import EnvironBase, EnvironConfig
from .maze import MazeGame
from .maze.utils import get_maze_env_obs
from sapien import Pose
from robotics.sim import Simulator


class WallConfig(EnvironConfig):
    #base_pose: PoseConfig = PoseConfig()
    maze_type: str = 'regular'
    width: int = 4
    height: int = 4
    maze_id: Optional[int] =None
    maze_height: float = 0.5
    wall_size: float = 0.025
    maze_size_scaling: float =4.


class SapienWallEnv(EnvironBase):
    def __init__(
        self, config: Optional[WallConfig] = None,
    ):
        self.config = config or WallConfig()
        self.base_pose = Pose(p=self.config.p, q=self.config.q)

    def reset_maze(self, maze_id=None):
        import random
        state = None
        maze_id = maze_id or self.config.maze_id
        if maze_id is not None:
            state = random.getstate()
            random.seed(maze_id)
        self.maze = MazeGame(self.config.height, width=self.config.width)
        if state is not None:
            random.setstate(state)

    def index2loc(self, i, j):
        loc = np.array([i, j]) + 0.5
        return loc * self.MAZE_SIZE_SCALING

    def loc2index(self, loc):
        return (loc / self.MAZE_SIZE_SCALING - 0.5).astype(np.int32)


    def _load(self, world: Simulator):
        scene = world._scene

        self.MAZE_HEIGHT = self.config.maze_height
        self.MAZE_SIZE_SCALING = self.config.maze_size_scaling
        self.width = self.config.width
        self.height = self.config.height
        maze_type = self.config.maze_type
        self.wall_size = self.config.wall_size

        self.objects = {}
        self.init_pose = {}

        self.ORIGIN_DELTA = np.array([self.width/2, self.height/2, 0.]) * self.MAZE_SIZE_SCALING 

        if maze_type == 'regular':
            for i in range(self.width+1):
                for j in range(self.height+1):
                    if i != self.width:
                        self.add_wall(scene, f"wall_{i}_{j}_h", i, j, 0)
                    if j != self.height:
                        self.add_wall(scene, f"wall_{i}_{j}_v", i, j, 1)
        else:
            raise NotImplementedError

        self.reset_maze()
        map = get_maze_env_obs(self.maze, 4)
        self.set_map(map)
        

    def add_wall(self, scene: sapien.Scene, name: str, x: int, y: int, dir: int):
        scale = self.MAZE_SIZE_SCALING

        if dir == 0:
            pos = ((x + 0.5) * scale, y * scale, self.MAZE_HEIGHT / 2 * scale)
            size = ((0.5 + self.wall_size) * scale, self.wall_size * scale, self.MAZE_HEIGHT / 2 * scale)
        else:
            pos = (x * scale, (y + 0.5) * scale, self.MAZE_HEIGHT / 2 * scale)
            size = (self.wall_size * scale, (0.5 + self.wall_size) * scale, self.MAZE_HEIGHT / 2 * scale)

        actor_builder = scene.create_actor_builder()
        actor_builder.add_box_collision(half_size=np.array(size))
        actor_builder.add_box_visual(half_size=np.array(size))
        box = actor_builder.build_static(name=name)
        box.set_pose(self.base_pose * sapien.Pose(p=np.array(pos) - self.ORIGIN_DELTA))

        self.objects[name] = box
        self.init_pose[name] = box.get_pose()

        return box


    def set_map(self, map):
        maze_type = self.config.maze_type
        if maze_type == 'regular' or maze_type is None:
            for i in range(self.width):
                for j in range(self.height):
                    if not map[0, j, i]:
                        name = f"wall_{i}_{j}_h"
                        #id = self.wrapped_env.model.geom_name2id(name)
                        init_pos = self.init_pose[name]
                        p = np.array(init_pos.p)
                        p[2] += 2000
                        self.objects[name].set_pose(sapien.Pose(p=p))
                    if not map[2, j, i]:
                        name = f"wall_{i}_{j}_v"
                        init_pos = self.init_pose[name]
                        p = np.array(init_pos.p)
                        p[2] += 2000
                        self.objects[name].set_pose(sapien.Pose(p=p))


    def _get_state(self):
        raise NotImplementedError

    def _set_state(self, state):
        raise NotImplementedError

    def _get_sapien_entity(self):
        return list(self.objects.values())