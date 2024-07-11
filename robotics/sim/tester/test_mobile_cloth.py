import argparse
import sapien
import numpy as np
from robotics.sim import Simulator, SimulatorConfig, CameraConfig
from robotics.sim.environs.ycb_clutter import YCBClutter, YCBConfig
from sapien import Pose

from robotics.sim.robot.mobile_sapien import MobileSapien
from robotics.sim.robot.mycobot280pi import MyCobot280Arm
from robotics.sim.ros_plugins.module import ROSModule


parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='thor', choices=['wall', 'thor', 'maze'])
parser.add_argument('--robot', type=str, default='large', choices=['small', 'large'])
parser.add_argument('--ros', action='store_true')
args = parser.parse_args()


# two types of robot
if args.robot == 'large':
    robot = MobileSapien(control_freq=50)
else:
    robot = MyCobot280Arm(50, arm_controller='posvel')

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


# add ycb clutter to the scene
# ycb = YCBClutter(YCBConfig()) 
element = {"scene": scene} #, 'ycb': ycb}

# create ros module. this will enable the ros plugins in the robot
ros = ROSModule('mobile_sapien', use_sim_time=False) if args.ros else None

# create simulator
sim = Simulator(
    SimulatorConfig(
        viewer_camera=CameraConfig(look_at=(-1., -0.2, 0.2), p=(0., 0, 1.8)),
        solver_iterations=50, sim_freq=100, enable_pcm=False), 
    robot, element, ros_module=ros, 
    add_cloth=True, cloth_init_pose=sapien.Pose([-3.79, -0.747, 1.2])
)

sim.reset()
robot.set_base_pose([-3.79, -0.747], 1.72)
robot.articulation.set_qvel(np.zeros(robot.articulation.dof, dtype=np.float32))

scene = sim._scene
cuboidsofabuilder = scene.create_actor_builder()
cuboidsofabuilder.add_convex_collision_from_file(filename="../assets/rigid/cuboidsofa/cuboidsofa.obj",scale=[0.55,0.55,0.55])
cuboidsofabuilder.add_visual_from_file(filename="../assets/rigid/cuboidsofa.glb",scale=[0.55,0.55,0.55])
cuboidsofa1=cuboidsofabuilder.build(name="cuboidsofa1")

engine = sim._engine
engine.set_pose(cuboidsofa1,Pose([-2.4, 0, 0.5], [1, 1,  -2.5, -2.5]))

idx=0
while not sim.viewer.closed:
    # action = np.zeros(robot.action_space.shape)
    idx += 1
    sim.viewer.paused = True
    sim.step(None) # you can pass action here to control the robot
    # if idx == 1:
    #     sim._scene.step()
    #     print(sim._scene.get_contacts())
    sim.render()
