from sapien import Pose
import numpy as np
import torch
from robotics.sim import Simulator, GPUEngineConfig, SimulatorConfig
from robotics.utils.sapien_utils import get_rigid_dynamic_component

from robotics.sim.robot.mycobot280pi import MyCobot280Arm  

robot = MyCobot280Arm(40, add_camera=False)
simulator = Simulator(
    SimulatorConfig(
        gpu_config=GPUEngineConfig()
    ),
    robot, {}
)
simulator.reset(init_engine=False) #initialize the simulator stuffs ..


engine = simulator._engine
engine.reset()

engine.set_root_pose(robot.articulation, Pose([1., 0, 0.], [1, 0, 0, 0]))

tot = 0
import tqdm
#while not simulator.viewer.closed:
for i in tqdm.tqdm(range(10000)):
    tot += 1
    action = np.zeros(robot.action_space.shape)
    action[0] = 1.
    action[-1] = 1. if (i//10)%2 == 0 else -1.
    simulator.step(action)
    simulator.render()