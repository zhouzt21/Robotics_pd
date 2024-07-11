from sapien import Pose
import torch
from robotics.sim import Simulator, GPUEngineConfig, SimulatorConfig
from robotics.utils.sapien_utils import get_rigid_dynamic_component

simulator = Simulator(
    SimulatorConfig(gpu_config=GPUEngineConfig()), # pass the GPUEngineConfig to enable the GPU engine
    None, {}
)

simulator.reset(init_engine=False) #initialize the simulator stuffs ..
scene = simulator._scene

actor_builder = scene.create_actor_builder()
actor_builder.add_box_collision(half_size=(0.1, 0.1, 0.1))
actor_builder.add_box_visual(half_size=(0.1, 0.1, 0.1), material=(1, 0, 0, 1))
actor = actor_builder.build()

actor_builder2 = scene.create_actor_builder()
actor_builder2.add_box_collision(half_size=(0.1, 0.1, 0.1))
actor_builder2.add_box_visual(half_size=(0.1, 0.1, 0.1), material=(0, 1, 0, 1))
actor2 = actor_builder2.build()

engine = simulator._engine

#rigid_comp = get_rigid_dynamic_component(actor)
#assert rigid_comp is not None
engine.set_pose(actor, Pose([0, 0, 2.], [1, 0, 0, 0]))
engine.set_pose(actor2, Pose([0, 1., 10.], [1, 0, 0, 0]))

engine.reset()

tot = 0
while not simulator.viewer.closed:
    tot += 1
    if tot % 100 == 0:
        engine.set_pose(actor, Pose([0, 0, 2.], [1, 0, 0, 0]))
        engine.set_pose(actor2, Pose([0, 1., 10.], [1, 0, 0, 0]))
    simulator.step(None)
    print(tot, engine.get_pose(actor).p[2], engine.get_pose(actor2).p[2])
    simulator.render()