import numpy as np
import open3d as o3d
import cv2
from robotics.sim import Simulator, SimulatorConfig, CameraConfig, MobileSmallRobot
#from realbot.sim.environs import SapienWallEnv, WallConfig
from robotics.sim.environs.ycb_clutter import YCBClutter, YCBConfig
from robotics.utils import tile_images

from robotics.sim.environs.thor import ThorEnv, ThorEnvConfig
from robotics.sim.robot.turtlebot4 import Turtlebot4

robot = Turtlebot4(control_freq=20)


scene = ThorEnv(ThorEnvConfig())
ycb = YCBClutter(YCBConfig()) 

from robotics.ros import ROSNode

node = ROSNode('turtlebot4')

sim = Simulator(
    SimulatorConfig(viewer_camera=CameraConfig(look_at=(-1., -0.2, 0.2), p=(0., 0, 1.8)),), 
    robot, {'scene': scene, 'ycb': ycb}, ros_node=node
)

sim.reset()
robot.set_base_pose([-1., -0.2], -1.67)


images = []
while True:
    sim.step(np.zeros(sim.action_space.shape))
    sim.render()
    if sim._viewer.closed:
        break