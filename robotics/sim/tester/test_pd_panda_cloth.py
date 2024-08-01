import argparse
import sapien
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from robotics.sim import PDSimConfig
from robotics.sim.pdcloth_env import PDClothEnv
from sapien import Pose

from robotics.sim.ros_plugins.module import ROSModule

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='thor', choices=['wall', 'thor', 'maze'])
# parser.add_argument('--robot', type=str, default='large', choices=['small', 'large'])
parser.add_argument('--ros', action='store_true')
args = parser.parse_args()

# various scenes
if args.scene == 'thor':
    from robotics.sim.environs.thor import ThorEnv, ThorEnvConfig
    scene = ThorEnv(ThorEnvConfig())
elif args.scene == 'wall':
    from robotics.sim.environs.sapien_square_wall import WallConfig, SquaWall
    scene = SquaWall(WallConfig())
else:
    from robotics.sim.environs.sapien_wall import SapienWallEnv, WallConfig
    scene = SapienWallEnv(WallConfig())

element = {"scene": scene} 

# create ros module. this will enable the ros plugins in the robot
ros = ROSModule('mobile_sapien', use_sim_time=False) if args.ros else None
# robot.set_base_pose([-3.79, -0.747], 1.72)
# robot.articulation.set_qvel(np.zeros(robot.articulation.dof, dtype=np.float32))

env = PDClothEnv( env_cfg= PDSimConfig( solver_iterations=50, sim_freq=100, enable_pcm=False, ros_module=ros, 
                                       render_mode="human", control_mode="pd_joint_pos"), 
                elements =element,  robot_state = {"robot_base_pose":(torch.FloatTensor([5, 0]), torch.FloatTensor([1.72])),
                                                   "robot_root_pose": Pose([2.5, 1.0, 0.5], [1, 0, 0, 0])
                                                   })
    
env.reset()

while True:
    env.render()
    env.viewer.paused = True
    env.step(None)    

