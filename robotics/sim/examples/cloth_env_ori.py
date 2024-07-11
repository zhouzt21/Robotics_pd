# type: ignore
from dataclasses import dataclass
from typing import Any

import dacite
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from sapienpd.pd_component import PDBodyComponent, PDClothComponent
from sapienpd.pd_config import PDConfig
from sapienpd.pd_defs import ShapeTypes
from sapienpd.pd_system import PDSystem

# pip install https://github.com/fbxiang/mesh2nvdb/releases/download/nightly/mesh2nvdb-0.1-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# pip install git+https://github.com/Rabbit-Hu/sapienpd


def gen_grid_cloth(size=(2, 1), resolution=(21, 11)):
    dim_x, dim_y = resolution
    xs = np.arange(dim_x)
    ys = np.arange(dim_y)
    row = np.vstack([xs[:-1], xs[1:], xs[1:] - dim_x, xs[:-1], xs[1:] - dim_x, xs[:-1] - dim_x])
    faces = (row[..., None] + ys[1:] * dim_x).reshape(6, -1).T.reshape(-1, 3)
    vertices = (np.stack(np.meshgrid(xs, ys), -1) * np.array(size) / (np.array(resolution) - 1)).reshape(-1, 2)
    vertices = np.pad(vertices, ((0, 0), (0, 1)))
    return vertices, faces


@dataclass
class FEMConfig:
    warp_device = "cuda:0"

    # memory config
    max_particles: int = 1 << 20
    max_constraints: int = 1 << 20
    max_constraint_total_size: int = 1 << 20
    max_colliders: int = 1 << 20

    # solver config
    sim_freq = 500
    pd_iterations = 20

    # physics config
    collision_margin = 0.2e-3
    collision_sphere_radius = 1.6e-3
    max_particle_velocity = 0.1


@register_env("PickCloth-v0")
class PickClothEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        fem_cfg: FEMConfig | dict = FEMConfig(),
        interaction_links=("panda_rightfinger", "panda_leftfinger"),
        cloth_size=(0.2, 0.2),
        cloth_resolution=(51, 51),
        cloth_init_pose=sapien.Pose([0.0, -0.1, 0.1]),
        robot_init_qpos_noise=0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.interaction_links = set(interaction_links)
        self.cloth_size = cloth_size
        self.cloth_resolution = cloth_resolution
        self.cloth_init_pose = cloth_init_pose

        if isinstance(fem_cfg, FEMConfig):
            self._fem_cfg = fem_cfg
        else:
            self._fem_cfg = dacite.from_dict(data_class=FEMConfig, data=fem_cfg, config=dacite.Config(strict=True))

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        assert (
            self._fem_cfg.sim_freq // self.sim_freq * self.sim_freq == self._fem_cfg.sim_freq
        ), "sim_freq must be constant multiple of sim_freq"

    def _setup_scene(self):
        super()._setup_scene()

        self._pd_config = PDConfig()
        self._pd_config.max_particles = self._fem_cfg.max_particles
        self._pd_config.max_constraints = self._fem_cfg.max_constraints
        self._pd_config.max_constraint_sizes_sum = self._fem_cfg.max_constraint_total_size
        self._pd_config.max_colliders = self._fem_cfg.max_colliders
        self._pd_config.time_step = 1 / self._fem_cfg.sim_freq
        self._pd_config.n_pd_iters = self._fem_cfg.pd_iterations
        self._pd_config.collision_margin = self._fem_cfg.collision_margin
        self._pd_config.collision_sphere_radius = self._fem_cfg.collision_sphere_radius
        self._pd_config.max_velocity = self._fem_cfg.max_particle_velocity
        self._pd_config.gravity = self.sim_cfg.scene_cfg.gravity

        self._pd_system = PDSystem(self._pd_config, self._fem_cfg.warp_device)
        assert len(self.scene.sub_scenes) == 1, "currently only single scene is supported"

        for s in self.scene.sub_scenes:
            s.add_system(self._pd_system)

        self._pd_ground = PDBodyComponent(
            [ShapeTypes.GEO_PLANE],
            frictions=[1.0],
            shape2cm=[sapien.Pose(q=[0.7071068, 0, -0.7071068, 0])],
        )
        entity = sapien.Entity()
        entity.add_component(self._pd_ground)
        self.scene.sub_scenes[0].add_entity(entity)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    def _load_scene(self, options: dict):
        # self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        # self.table_scene.build()

        # b = self.scene.create_actor_builder()
        # b.add_multiple_convex_collisions_from_file(
        #     "assets/banana/collision_meshes/collision.obj"
        # )
        # b.add_visual_from_file(
        #     "assets/banana/visual_meshes/visual.glb"
        # )
        # banana = b.build(name="banana")
        # banana.set_pose(sapien.Pose(p=[0.1, 0, 0.02]))
        # banana._objs[0].add_component(
        #     PDBodyComponent.from_physx_shape(
        #         banana._objs[0].find_component_by_type(
        #             sapien.pysapien.physx.PhysxRigidDynamicComponent
        #         ),
        #         grid_size=1e-3,
        #     )
        # )
        # self.banana = banana

        for s in self.scene.sub_scenes:
            for e in s.entities:
                if e.name not in self.interaction_links:
                    continue
                body = e.find_component_by_type(sapien.pysapien.physx.PhysxRigidBodyComponent)
                e.add_component(PDBodyComponent.from_physx_shape(body, grid_size=3e-3))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # with torch.device(self.device):
        #     self.table_scene.initialize(env_idx)

        self.set_pd_state_dict(self.pd_init_state)

    def _after_reconfigure(self, options):
        vertices, faces = gen_grid_cloth(size=self.cloth_size, resolution=self.cloth_resolution)

        cloth_comp = PDClothComponent(
            vertices,
            faces,
            thickness=1e-3,
            density=1e3,
            stretch_stiffness=1e3,
            bend_stiffness=1e-3,
            friction=0.3,
            collider_iterpolation_depth=0,
        )
        cloth_render = sapien.render.RenderCudaMeshComponent(len(vertices), 2 * len(faces))
        cloth_render.set_vertex_count(len(vertices))
        cloth_render.set_triangle_count(2 * len(faces))
        cloth_render.set_triangles(np.concatenate([faces, faces[:, ::-1]], axis=0))
        cloth_render.set_material(sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.4, 1.0]))
        cloth_entity = sapien.Entity()
        cloth_entity.add_component(cloth_comp)
        cloth_entity.add_component(cloth_render)
        cloth_entity.set_pose(self.cloth_init_pose)

        self.scene.sub_scenes[0].add_entity(cloth_entity)
        self.cloth_comp = cloth_comp
        self.cloth_render_comp = cloth_render

        self.pd_init_state = self.get_pd_state_dict()

    def render_human(self):
        self.cloth_comp.update_render(self.cloth_render_comp)
        return super().render_human()

    def render(self):
        self.cloth_comp.update_render(self.cloth_render_comp)
        return super().render()

    def _after_simulation_step(self):
        self._pd_system.sync_body()
        for _ in range(self._fem_cfg.sim_freq // self.sim_freq):
            self._pd_system.step()

    def get_pd_state_dict(self):
        return {
            "q": self._pd_system.q.numpy(),
            "qd": self._pd_system.qd.numpy(),
            "body_q": self._pd_system.body_q.numpy(),
        }

    def set_pd_state_dict(self, state):
        self._pd_system.q.assign(state["q"])
        self._pd_system.qd.assign(state["qd"])
        self._pd_system.body_q.assign(state["body_q"])

    def get_state_dict(self):
        return super().get_state_dict() | {"pd": self.get_pd_state_dict()}

    def set_state_dict(self, state):
        super().set_state_dict(state)
        self.set_pd_state_dict(state["pd"])


def main():
    env = PickClothEnv(render_mode="human", control_mode="pd_joint_pos")
    num_trajs = 0
    seed = 0
    env.reset(seed=seed)
    while True:
        print(f"Collecting trajectory {num_trajs + 1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue
        elif code == "restart":
            env.reset(seed=seed, options={"save_trajectory": False})


def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_acc_limits=0.5,
        joint_vel_limits=0.5,
    )
    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True
    viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "panda_hand")._objs[0].entity)
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
    while True:
        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost panda arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data and save videos
            """)
            pass
        # elif viewer.window.key_press("k"):
        #     print("Saving checkpoint")
        #     last_checkpoint_state = env.get_state_dict()
        # elif viewer.window.key_press("l"):
        #     if last_checkpoint_state is not None:
        #         print("Loading previous checkpoint")
        #         env.set_state_dict(last_checkpoint_state)
        #     else:
        #         print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        # elif viewer.window.key_press("r"):
        #     viewer.select_entity(None)
        #     return "restart"
        # elif viewer.window.key_press("t"):
        #     # TODO (stao): change from position transform to rotation transform
        #     pass
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g"):
            if gripper_open:
                gripper_open = False
                _, reward, _, _, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _, _, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        # # TODO left, right depend on orientation really.
        # elif viewer.window.key_press("down"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, 0, 0.01]))
        # elif viewer.window.key_press("up"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, 0, -0.01]))
        # elif viewer.window.key_press("right"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, -0.01, 0]))
        # elif viewer.window.key_press("left"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, +0.01, 0]))
        if execute_current_pose:
            # z-offset of end-effector gizmo to TCP position is hardcoded for the panda robot here
            result = planner.move_to_pose_with_screw(
                transform_window._gizmo_pose * sapien.Pose([0, 0, 0.102]), dry_run=True
            )
            if result != -1 and len(result["position"]) < 100:
                _, reward, _, _, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1:
                    print("Plan failed")
                else:
                    print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False


if __name__ == "__main__":
    main()
