# python3 tester/test_mobile_sapien.py --scene thor --robot large --ros

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
    SimulatorConfig(viewer_camera=CameraConfig(look_at=(-1., -0.2, 0.2), p=(0., 0, 1.8)),solver_iterations=50, sim_freq=100, enable_pcm=False), 
    robot, element, ros_module=ros
)

# disable the depth camera
# if ros is not None:
#     for i in ros.get_active_ros_plugins():
#         if i.__class__.__name__ == 'RGBDPublisher':
#             i.enabled = False

# must reset the simulator before using it. this will load the scene and the robot.
sim.reset()
robot.set_base_pose([-3.79, -0.747], 1.72)
robot.articulation.set_qvel(np.zeros(robot.articulation.dof, dtype=np.float32))

# TURN ON this so that the footprint is updated
# module.update_footprint(g_radius=False)
scene = sim._scene
# load sapien partnet urdf (scale the shelf)
loader = scene.create_urdf_loader()
loader.scale = 0.6

sofabuilder = scene.create_actor_builder()
sofabuilder.add_convex_collision_from_file(filename="../asset/rigid/sofa/sofa.obj",scale=[1,1,1])
sofabuilder.add_visual_from_file(filename="../asset/rigid/sofa.glb",scale=[1,1,1])
sofa1=sofabuilder.build(name="sofa1")
sofa2=sofabuilder.build(name="sofa2")
sofa3=sofabuilder.build(name="sofa3")

chairbuilder = scene.create_actor_builder()
chairbuilder.add_convex_collision_from_file(filename="../asset/rigid/chair/chair.obj",scale=[0.55,0.55,0.55])
chairbuilder.add_visual_from_file(filename="../asset/rigid/chair.glb",scale=[0.55,0.55,0.55])
chair1_1=chairbuilder.build(name="chair1_1")
chair1_2=chairbuilder.build(name="chair1_2")
chair1_3=chairbuilder.build(name="chair1_3")
chair1_4=chairbuilder.build(name="chair1_4")
chair1_5=chairbuilder.build(name="chair1_5")
chair1_6=chairbuilder.build(name="chair1_6")
chair1_7=chairbuilder.build(name="chair1_7")
chair1_8=chairbuilder.build(name="chair1_8")
chair1_9=chairbuilder.build(name="chair1_9")
chair1_10=chairbuilder.build(name="chair1_10")
chair1_11=chairbuilder.build(name="chair1_11")
chair1_12=chairbuilder.build(name="chair1_12")

chair2_1=chairbuilder.build(name="chair2_1")
chair2_2=chairbuilder.build(name="chair2_2")
chair2_3=chairbuilder.build(name="chair2_3")
chair2_4=chairbuilder.build(name="chair2_4")
chair2_5=chairbuilder.build(name="chair2_5")
chair2_6=chairbuilder.build(name="chair2_6")
chair2_7=chairbuilder.build(name="chair2_7")
chair2_8=chairbuilder.build(name="chair2_8")
chair2_9=chairbuilder.build(name="chair2_9")
chair2_10=chairbuilder.build(name="chair2_10")
chair2_11=chairbuilder.build(name="chair2_11")
chair2_12=chairbuilder.build(name="chair2_12")

tablebuilder = scene.create_actor_builder()
tablebuilder.add_convex_collision_from_file(filename="../asset/rigid/table/table.obj",scale=[0.7,0.7,0.7])
tablebuilder.add_visual_from_file(filename="../asset/rigid/table.glb",scale=[0.7,0.7,0.7])
table1=tablebuilder.build(name="table1")
table2=tablebuilder.build(name="table2")
table3=tablebuilder.build(name="table3")
table4=tablebuilder.build(name="table4")
table5=tablebuilder.build(name="table5")
table6=tablebuilder.build(name="table6")
table7=tablebuilder.build(name="table7")
table8=tablebuilder.build(name="table8")

cotablebuilder = scene.create_actor_builder()
cotablebuilder.add_convex_collision_from_file(filename="../asset/rigid/coffeetable/coffeetable.obj",scale=[0.7,0.7,0.7])
cotablebuilder.add_visual_from_file(filename="../asset/rigid/coffeetable.glb",scale=[0.7,0.7,0.7])
cotable=cotablebuilder.build(name="coffeetable")

cubesofabuilder = scene.create_actor_builder()
cubesofabuilder.add_convex_collision_from_file(filename="../asset/rigid/cubesofa/cubesofa.obj",scale=[0.35,0.35,0.35])
cubesofabuilder.add_visual_from_file(filename="../asset/rigid/cubesofa.glb",scale=[0.35,0.35,0.35])
cubesofa1_1=cubesofabuilder.build(name="cubesofa1_1")
cubesofa1_2=cubesofabuilder.build(name="cubesofa1_2")
cubesofa1_3=cubesofabuilder.build(name="cubesofa1_3")
cubesofa1_4=cubesofabuilder.build(name="cubesofa1_4")
cubesofa1_5=cubesofabuilder.build(name="cubesofa1_5")
cubesofa1_6=cubesofabuilder.build(name="cubesofa1_6")
cubesofa1_7=cubesofabuilder.build(name="cubesofa1_7")
cubesofa1_8=cubesofabuilder.build(name="cubesofa1_8")
cubesofa1_9=cubesofabuilder.build(name="cubesofa1_9")
cubesofa1_10=cubesofabuilder.build(name="cubesofa1_10")
cubesofa1_11=cubesofabuilder.build(name="cubesofa1_11")

cubesofa2_1=cubesofabuilder.build(name="cubesofa2_1")
cubesofa2_2=cubesofabuilder.build(name="cubesofa2_2")
cubesofa2_3=cubesofabuilder.build(name="cubesofa2_3")
cubesofa2_4=cubesofabuilder.build(name="cubesofa2_4")
cubesofa2_5=cubesofabuilder.build(name="cubesofa2_5")
cubesofa2_6=cubesofabuilder.build(name="cubesofa2_6")
cubesofa2_7=cubesofabuilder.build(name="cubesofa2_7")
cubesofa2_8=cubesofabuilder.build(name="cubesofa2_8")
cubesofa2_9=cubesofabuilder.build(name="cubesofa2_9")
cubesofa2_10=cubesofabuilder.build(name="cubesofa2_10")
cubesofa2_11=cubesofabuilder.build(name="cubesofa2_11")

cubesofa3_1=cubesofabuilder.build(name="cubesofa3_1")
cubesofa3_2=cubesofabuilder.build(name="cubesofa3_2")
cubesofa3_3=cubesofabuilder.build(name="cubesofa3_3")
cubesofa3_4=cubesofabuilder.build(name="cubesofa3_4")
cubesofa3_5=cubesofabuilder.build(name="cubesofa3_5")
cubesofa3_6=cubesofabuilder.build(name="cubesofa3_6")
cubesofa3_7=cubesofabuilder.build(name="cubesofa3_7")
cubesofa3_8=cubesofabuilder.build(name="cubesofa3_8")
cubesofa3_9=cubesofabuilder.build(name="cubesofa3_9")
cubesofa3_10=cubesofabuilder.build(name="cubesofa3_10")
cubesofa3_11=cubesofabuilder.build(name="cubesofa3_11")

cuboidsofabuilder = scene.create_actor_builder()
cuboidsofabuilder.add_convex_collision_from_file(filename="../asset/rigid/cuboidsofa/cuboidsofa.obj",scale=[0.55,0.55,0.55])
cuboidsofabuilder.add_visual_from_file(filename="../asset/rigid/cuboidsofa.glb",scale=[0.55,0.55,0.55])
cuboidsofa1=cuboidsofabuilder.build(name="cuboidsofa1")
cuboidsofa2=cuboidsofabuilder.build(name="cuboidsofa2")
cuboidsofa3=cuboidsofabuilder.build(name="cuboidsofa3")
cuboidsofa4=cuboidsofabuilder.build(name="cuboidsofa4")
cuboidsofa5=cuboidsofabuilder.build(name="cuboidsofa5")
cuboidsofa6=cuboidsofabuilder.build(name="cuboidsofa6")
cuboidsofa7=cuboidsofabuilder.build(name="cuboidsofa7")
cuboidsofa8=cuboidsofabuilder.build(name="cuboidsofa8")

engine = sim._engine
engine.set_pose(sofa1, Pose([2, 2.7, 0.5], [1, 1, 0.8, 0.8]))
engine.set_pose(sofa2, Pose([3.5, 1.8, 0.5], [1, 1, 3.5, 3.5]))
engine.set_pose(sofa3, Pose([4.6, 3, 0.5], [1, 1, -2, -2])) # 4 3.5
engine.set_pose(cotable,Pose([3.3, 3, 0.5], [1, 1, 0, 0]))

#-------- table and chair -------

engine.set_pose(table1,Pose([-0.6, 3.7, 0.5], [1, 1, -1.8, -1.8]))
engine.set_pose(table2,Pose([-0.6, 2.7, 0.5], [1, 1, -1.8, -1.8]))
engine.set_pose(table3,Pose([-1.2, 3.7, 0.5], [1, 1, -1.8, -1.8]))
engine.set_pose(table4,Pose([-1.2, 2.7, 0.5], [1, 1, -1.8, -1.8]))

engine.set_pose(chair1_1, Pose([0, 4, 0.5],  [1, 1, -0.5 ,-0.5]))
engine.set_pose(chair1_2, Pose([0, 3.5, 0.5],[1, 1, -0.5 ,-0.5]))
engine.set_pose(chair1_3, Pose([0, 3, 0.5],  [1, 1, -0.5 ,-0.5]))
engine.set_pose(chair1_4, Pose([0, 2.5, 0.5],[1, 1, -0.5 ,-0.5]))
engine.set_pose(chair1_5, Pose([-1.8, 4, 0.5],  [1, 1, 2 ,2]))
engine.set_pose(chair1_6, Pose([-1.8, 3.5, 0.5],[1, 1, 2 ,2]))
engine.set_pose(chair1_7, Pose([-1.8, 3, 0.5],  [1, 1, 2 ,2]))
engine.set_pose(chair1_8, Pose([-1.8, 2.5, 0.5],[1, 1, 2 ,2]))
engine.set_pose(chair1_9, Pose([-0.6, 1.8, 0.5],[1, 1, -2.5 ,-2.5]))
engine.set_pose(chair1_10, Pose([-1.2, 1.8, 0.5],[1, 1, -2.5 ,-2.5]))
engine.set_pose(chair1_11, Pose([-0.6, 4.6, 0.5],[1, 1, 0.4 ,0.4]))
engine.set_pose(chair1_12, Pose([-1.2, 4.6, 0.5],[1, 1, 0.4 ,0.4]))


engine.set_pose(table5,Pose([-3, 3.7, 0.5], [1, 1, -1.8, -1.8]))
engine.set_pose(table6,Pose([-3, 2.7, 0.5], [1, 1, -1.8, -1.8]))
engine.set_pose(table7,Pose([-3.6, 3.7, 0.5], [1, 1, -1.8, -1.8]))
engine.set_pose(table8,Pose([-3.6, 2.7, 0.5], [1, 1, -1.8, -1.8]))

engine.set_pose(chair2_1, Pose([-2.4, 4,  0.5],[1, 1, -0.5 ,-0.5]))
engine.set_pose(chair2_2, Pose([-2.4, 3.5, 0.5],[1, 1, -0.5 ,-0.5]))
engine.set_pose(chair2_3, Pose([-2.4, 3,  0.5],[1, 1, -0.5 ,-0.5]))
engine.set_pose(chair2_4, Pose([-2.4, 2.5, 0.5],[1, 1, -0.5 ,-0.5]))
engine.set_pose(chair2_5, Pose([-4.2, 4,  0.5],[1, 1, 2 ,2]))
engine.set_pose(chair2_6, Pose([-4.2, 3.5, 0.5],[1, 1, 2 ,2]))
engine.set_pose(chair2_7, Pose([-4.2, 3,  0.5],[1, 1, 2 ,2]))
engine.set_pose(chair2_8, Pose([-4.2, 2.5, 0.5],[1, 1, 2 ,2]))
engine.set_pose(chair2_9, Pose([-3, 1.8, 0.5],[1, 1, -2.5 ,-2.5]))
engine.set_pose(chair2_10, Pose([-3.6, 1.8, 0.5],[1, 1, -2.5 ,-2.5]))
engine.set_pose(chair2_11, Pose([-3, 4.6, 0.5],[1, 1, 0.4 ,0.4]))
engine.set_pose(chair2_12, Pose([-3.6, 4.6, 0.5],[1, 1, 0.4 ,0.4]))


#----- cubesofa -------

engine.set_pose(cubesofa1_1,Pose([0.4, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_2,Pose([0, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_3,Pose([-0.4, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_4,Pose([-0.8, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_5,Pose([-1.2, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_6,Pose([-1.6, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_7,Pose([-2, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_8,Pose([-2.4, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_9,Pose([-2.8, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_10,Pose([-3.2, -1.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa1_11,Pose([-3.6, -1.5, 0.5], [1, 1, -3.8, -3.8]))

engine.set_pose(cubesofa2_1,Pose([0.4, -3, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_2,Pose([0, -3,0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_3,Pose([-0.4, -3, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_4,Pose([-0.8, -3, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_5,Pose([-1.2, -3, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_6,Pose([-1.6, -3, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_7,Pose([-2, -3,0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_8,Pose([-2.4, -3, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_9,Pose([-2.8, -3, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_10,Pose([-3.2,-3, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa2_11,Pose([-3.6,-3, 0.5], [1, 1, -3.8, -3.8]))

engine.set_pose(cubesofa3_1,Pose([0.4, -4.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_2,Pose([0, -4.5,0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_3,Pose([-0.4, -4.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_4,Pose([-0.8, -4.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_5,Pose([-1.2, -4.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_6,Pose([-1.6, -4.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_7,Pose([-2, -4.5,0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_8,Pose([-2.4, -4.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_9,Pose([-2.8, -4.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_10,Pose([-3.2,-4.5, 0.5], [1, 1, -3.8, -3.8]))
engine.set_pose(cubesofa3_11,Pose([-3.6,-4.5, 0.5], [1, 1, -3.8, -3.8]))

#---cubiodsofa---

engine.set_pose(cuboidsofa1,Pose([0.6, 0, 0.5], [1, 1, -2.5, -2.5]))
engine.set_pose(cuboidsofa2,Pose([0, 0, 0.5], [1, 1, -2.5, -2.5]))
engine.set_pose(cuboidsofa3,Pose([-0.6, 0, 0.5], [1, 1,  -2.5, -2.5]))
engine.set_pose(cuboidsofa4,Pose([-1.2, 0, 0.5], [1, 1,  -2.5, -2.5]))
engine.set_pose(cuboidsofa5,Pose([-1.8, 0, 0.5], [1, 1,  -2.5, -2.5]))
engine.set_pose(cuboidsofa6,Pose([-2.4, 0, 0.5], [1, 1,  -2.5, -2.5]))
engine.set_pose(cuboidsofa7,Pose([-3, 0, 0.5], [1, 1, -2.5, -2.5]))
engine.set_pose(cuboidsofa8,Pose([-3.6, 0, 0.5], [1, 1, -2.5, -2.5]))


# ----- shelf -----
token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Inpob3UtenQyMUBtYWlscy50c2luZ2h1YS5lZHUuY24iLCJpcCI6IjE3Mi4yMC4wLjEiLCJwcml2aWxlZ2UiOjEsImZpbGVPbmx5Ijp0cnVlLCJpYXQiOjE3MTEwNzQ2OTcsImV4cCI6MTc0MjYxMDY5N30.9tSKDRTDFPZV27q3cHFQac_Yd36Q_aK2r70TThN-3YU'
urdf_file = sapien.asset.download_partnet_mobility(47238, token)
test_carto = sapien.asset.download_partnet_mobility(46033, token)

testshelf1 = loader.load(urdf_file)
testshelf2 = loader.load(urdf_file)
testshelf3 = loader.load(urdf_file)
testshelf4 = loader.load(urdf_file)

testshelf5 = loader.load(urdf_file)
testshelf6 = loader.load(urdf_file)
testshelf7 = loader.load(urdf_file)
testshelf8 = loader.load(urdf_file)
testshelf9 = loader.load(urdf_file)

testcartp = loader.load(test_carto)

engine.set_pose(testcartp, Pose([3.5, -4.56, 0.5], [1, 0, 0, 0]))

testshelf1.set_root_pose(Pose([5.4, -3.62, 0.5], [1, 0, 0, 0]))
testshelf2.set_root_pose(Pose([5.4, -2.68, 0.5], [1, 0, 0, 0]))
testshelf3.set_root_pose(Pose([5.4, -1.74, 0.5], [1, 0, 0, 0]))
testshelf4.set_root_pose(Pose([5.4, -0.8, 0.5], [1, 0, 0, 0]))

testshelf5.set_root_pose(Pose([4, -3.62, 0.5], [0, 0, 1, 0]))
testshelf6.set_root_pose(Pose([4, -2.68, 0.5], [0, 0, 1, 0]))
testshelf7.set_root_pose(Pose([4, -1.74, 0.5], [0, 0, 1, 0]))
testshelf8.set_root_pose(Pose([4, -0.8, 0.5], [0, 0, 1, 0]))
testshelf9.set_root_pose(Pose([4, 0.14, 0.5], [0, 0, 1, 0]))


engine.reset()
idx=0
images = []
while not sim.viewer.closed:
    # action = np.zeros(robot.action_space.shape)
    idx += 1
    sim.step(None) # you can pass action here to control the robot
    # if idx == 1:
    #     sim._scene.step()
    #     print(sim._scene.get_contacts())
    sim.render()