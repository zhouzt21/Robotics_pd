from sapien import Entity, Pose
import cv2
import numpy as np
from sapien.pysapien.physx import PhysxArticulation
from robotics.sim import Simulator, GPUEngineConfig, SimulatorConfig
import tqdm
# from robotics.utils.sapien_utils import get_rigid_dynamic_component

from robotics.sim.robot.mycobot280pi import MyCobot280Arm
from robotics.sim.sensors.camera_v2 import CameraV2, CameraConfig
from robotics.sim.environ import EnvironBase, EnvironConfig


N = 200

thetas = np.linspace(0, 2*np.pi, N + 1)[:N]
targets = np.stack([np.cos(thetas), np.sin(thetas)]).T

class Box(EnvironBase):
    def _load(self, world: Simulator):
        scene = world._scene
        actor_builder = scene.create_actor_builder()
        actor_builder.add_box_collision(half_size=(0.1, 0.1, 0.1))
        actor_builder.add_box_visual(half_size=(0.1, 0.1, 0.1), material=(1, 0, 0, 1))
        actor = actor_builder.build()
        self.actor = actor
        x, y = targets[world._scene_idx]
        world._engine.set_pose(actor, Pose([x, y + 5., 0.], [1, 0, 0, 0]))

    def _get_sapien_entity(self):
        return [self.actor]


robot = MyCobot280Arm(40, add_camera=False, arm_controller='vel')
box = Box(EnvironConfig())
camera = CameraV2(CameraConfig(look_at=(0., 0., 0.5), p=(1.5, 1.5, 2.)))

simulator = Simulator(
    SimulatorConfig(
        gpu_config=GPUEngineConfig(),
        n_scenes=N,
        viewer_camera=CameraConfig(look_at=(0., 0., 0.5), p=(3.5, 10.5, 5.5)),
    ),
    robot, {'box': box, 'camera': camera},
)
simulator.reset() #initialize the simulator stuffs ..

engine = simulator._engine
engine.set_root_pose(robot.articulation, Pose([0., 0, 0.], [1, 0, 0, 0]))

tot = 0


simulator.set_viewer_scenes(list(range(min(N, 30))))

for i in tqdm.tqdm(range(100)):
    tot += 1
    action = np.zeros((N, *robot.action_space.shape))
    action[..., -1] = 1. if (i//10) % 2 == 0 else -1.

    xy, theta = robot.get_base_pose()
    K = 10
    to = targets - xy[..., :2]

    angle_diff = (np.arctan2(to[:, 1], to[:, 0]) - theta + np.pi) % (2 * np.pi) - np.pi
    dist = np.linalg.norm(to, axis=-1) 

    action[..., 0] = K * dist * ( np.abs(angle_diff) < np.pi / 2 )
    action[..., 2] = K * angle_diff * ( dist > 0.01 )

    if N == 1:
        action = action[0]
    simulator.step(action)
    image = simulator._engine.take_picture(0)['camera']['Color']
    image2 = simulator._engine.take_picture(N//2)['camera']['Color']
    cv2.imshow('image', cv2.resize(np.concatenate([image, image2], axis=1), (0, 0), fx=0.25, fy=0.25)[..., [2, 1, 0]])
    cv2.waitKey(1)
    simulator.render()

print(dist)
print(theta)