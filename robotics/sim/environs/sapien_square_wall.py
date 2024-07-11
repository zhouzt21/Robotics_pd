from typing import Optional
import sapien.core as sapien
import numpy as np
from ..environ import EnvironBase, EnvironConfig
from .maze import MazeGame
from robotics.sim import Simulator


class WallConfig(EnvironConfig):
    maze_type: str = 'regular'
    width: int = 4
    height: int = 4
    maze_id: Optional[int] =None
    maze_height: float = 0.5
    wall_size: float = 0.025
    maze_size_scaling: float =4.


class SquaWall(EnvironBase):
    def __init__(
        self, config: Optional[WallConfig] = None,
    ):
        self.config = config or WallConfig()

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


    def _load(self, world: "Simulator"):
        scene = world._scene

        self.MAZE_HEIGHT = self.config.maze_height
        self.MAZE_SIZE_SCALING = self.config.maze_size_scaling
        self.width = 4
        self.height = 3
        maze_type = self.config.maze_type
        self.wall_size = self.config.wall_size

        self.objects = {}
        self.init_pose = {}

        self.ORIGIN_DELTA = np.array([self.width/2, self.height/2, 0.]) * self.MAZE_SIZE_SCALING 

        d,l,w,h = 6, 6, 0.4, 4
        actor_builder = scene.create_actor_builder()
        actor_builder.add_box_collision(half_size=np.array([w,l,h]))
        actor_builder.add_box_visual(half_size=np.array([w,l,h]))
        box = actor_builder.build_static(name="wall_right")
        box.set_pose(sapien.Pose(p=np.array([d,0,h])))
        name="wall_right"
        self.objects[name] = box
        self.init_pose[name] = box.get_pose()

        actor_builder.add_box_collision(half_size=np.array([w,l,h]))
        actor_builder.add_box_visual(half_size=np.array([w,l,h]))
        box2 = actor_builder.build_static(name="wall_left")
        box2.set_pose(sapien.Pose(p=np.array([-d,0,h])))
        name="wall_left"
        self.objects[name] = box
        self.init_pose[name] = box.get_pose()

        # Bottom Wall
        actor_builder.add_box_collision(half_size=np.array([w, l, h]))
        actor_builder.add_box_visual(half_size=np.array([w, l, h]))
        box3 = actor_builder.build_static(name="wall_bottom")
        box3.set_pose(sapien.Pose(p=np.array([0, -d, h]), q=np.array([0.707, 0, 0, 0.707])))
        name = "wall_bottom"
        self.objects[name] = box3
        self.init_pose[name] = box3.get_pose()

        # Top Wall
        actor_builder.add_box_collision(half_size=np.array([w, l, h]))
        actor_builder.add_box_visual(half_size=np.array([w, l, h]))
        box4 = actor_builder.build_static(name="wall_top")
        box4.set_pose(sapien.Pose(p=np.array([0, d, h]), q=np.array([0.707, 0, 0, 0.707])))
        name = "wall_top"
        self.objects[name] = box4
        self.init_pose[name] = box4.get_pose()

        self.reset_maze()

        

    def _get_state(self):
        raise NotImplementedError

    def _set_state(self, state):
        raise NotImplementedError

    def _get_sapien_entity(self):
        return list(self.objects.values())