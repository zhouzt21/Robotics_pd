import os
import numpy as np
import igl
from mesh2nvdb import mesh2nvdb
from PIL import Image
import warp as wp
import sapien

import sapienpd
from sapienpd.pd_config import PDConfig
from sapienpd.pd_system import PDSystem
from sapienpd.pd_defs import *
from sapienpd.pd_component import PDClothComponent, PDBodyComponent
from sapienpd.utils.array import wp_slice
from sapienpd.utils.logger import sapienpd_logger as logger


logger.setLevel("DEBUG")


def init_camera(scene: sapien.Scene):
    cam_entity = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(512, 512)
    cam.set_near(1e-3)
    cam.set_far(1000)
    cam_entity.add_component(cam)
    cam_entity.name = "camera"
    cam_entity.set_pose(
        sapien.Pose(
            [-0.064028, -0.270365, 0.440705], [0.73446, -0.146693, 0.1681, 0.64093]
        )
    )
    scene.add_entity(cam_entity)
    return cam


def init_ground(scene: sapien.Scene):
    # ground_type = "plane"
    ground_type = "sphere"
    # ground_type = "cow"
    # ground_type = "banana"
        
    if ground_type == "plane":
        ground_pose = sapien.Pose([0.1, 0.1, 0.0], [0.7071068, 0, -0.7071068, 0])
        # ground_pose = sapien.Pose([0.5, 0.5, 0.0], [0.82942284137, 0.10980760492, -0.53660256158, -0.10980760492])
        # ground_pose = sapien.Pose([0.5, 0.5, 0.0], [0.86602537634, 0, -0.50000002661, 0])
        # ground_n = ground_pose.to_transformation_matrix()[:3, :3] @ np.array([1.0, 0.0, 0.0])
        # ground_angle = np.arccos(ground_n[2])
        # print(f"ground_n: {ground_n}, angle: {ground_angle * 180 / np.pi}")
        # ground_pose = sapien.Pose([0.5, 0.5, 0.0], [ 0.7544065, 0.1330222, -0.6330222, -0.1116189 ])
        # ground_twist = np.array([0.1, 0.0, np.sqrt(3)*0.1, 0.0, 0.0, 0.0])
        ground_twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ground_comp = PDBodyComponent(
            shape_types=[ShapeTypes.GEO_PLANE],
            scales=np.array([[0.0, 1.0, 1.0]]),
            frictions=[1.0],
            cm2body=sapien.Pose([0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]), 
            shape2cm=[sapien.Pose([0.0, -1.0, -1.0], [1.0, 0.0, 0.0, 0.0])]
        )
        ground_render = sapien.render.RenderBodyComponent()
        material = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        material.set_diffuse_texture(sapien.render.RenderTexture2D("assets/tablecloth.jpg"))
        ground_render.attach(
            sapien.render.RenderShapePlane(
                np.array([1.0, 1.0, 1.0]),
                material
            )
        )

        ground_entity = sapien.Entity()
        ground_entity.add_component(ground_comp)
        ground_entity.add_component(ground_render)
        # ground_entity.set_pose(ground_pose)
        scene.add_entity(ground_entity)
        ground_comp.set_pose_twist(ground_pose, ground_twist)

    elif ground_type == "sphere":
        ground_pose = sapien.Pose([0.5, 0.5, -0.1], [0.7071068, 0, 0.7071068, 0])
        # ground_pose = sapien.Pose([0.5, 0.5, -0.1], [1, 0, 0, 0])
        ground_twist = np.array([0.0, 0.0, 0.2, 0.1, 0.0, 0.0])
        ground_comp = PDBodyComponent(
            shape_types=[ShapeTypes.GEO_SPHERE],
            scales=np.array([[0.5, 0.5, 0.5]]),
            frictions=[1.0],
        )
        ground_render = sapien.render.RenderBodyComponent()
        material = sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        material.set_diffuse_texture(sapien.render.RenderTexture2D("assets/earth.jpg"))
        ground_render.attach(
            sapien.render.RenderShapeSphere(
                # 0.5, sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
                0.5, material
            )
        )

        ground_entity = sapien.Entity()
        ground_entity.add_component(ground_comp)
        ground_entity.add_component(ground_render)
        # ground_entity.set_pose(ground_pose)
        scene.add_entity(ground_entity)
        ground_comp.set_pose_twist(ground_pose, ground_twist)

    elif ground_type == "cow":
        obj = os.path.join(os.path.dirname(__file__), "assets/spot.obj")
        scale = np.ones(3) * 0.5
        vertices, faces = igl.read_triangle_mesh(obj)
        vertices = vertices * 0.5
        nvdb = mesh2nvdb(vertices, faces, 0.01)
        volume = wp.Volume.load_from_nvdb(nvdb)
        ground_pose = sapien.Pose([0.5, 0.6, 0.0], [0.7071068, 0.7071068, 0.0, 0.0])
        ground_twist = np.array([0.0, 0.0, 0.5, 0.2, 0.0, 0.0])
        ground_comp = PDBodyComponent(
            shape_types=[ShapeTypes.GEO_SDF],
            # scale=scale,
            frictions=[4.0],
            volumes=[volume],
            shape2cm=[
                sapien.Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            ]
        )
        ground_render = sapien.render.RenderBodyComponent()
        ground_render.attach(
            sapien.render.RenderShapeTriangleMesh(
                obj, 
                scale=scale
            )
        )

        ground_entity = sapien.Entity()
        ground_entity.add_component(ground_comp)
        ground_entity.add_component(ground_render)
        # ground_entity.set_pose(ground_pose)
        scene.add_entity(ground_entity)
        ground_comp.set_pose_twist(ground_pose, ground_twist)

    elif ground_type == "banana":
        scale = np.ones(3) * 5.0
        ground_pose = sapien.Pose([0.5, 0.6, 0.0])
        ground_twist = np.array([0.0, 0.0, 0.5, 0.2, 0.0, 0.0])
        builder = scene.create_actor_builder()
        # builder.add_convex_collision_from_file(
        #     filename="assets/banana/collision_meshes/collision.obj"
        # )
        builder.add_multiple_convex_collisions_from_file(
            filename="../assets/banana/collision_meshes/collision.obj",
            scale=scale
        )
        # builder.add_visual_from_file(filename="assets/banana/visual_meshes/visual.dae")
        builder.add_visual_from_file(
            filename="../assets/banana/collision_meshes/collision.obj",
            scale=scale
        )
        ground_entity = builder.build(name="ground")
        ground_entity.set_pose(ground_pose)

        for comp in ground_entity.get_components():
            if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                rigid_comp = comp
                pd_comp = PDBodyComponent.from_physx_shape(comp, grid_size=2e-3)
                ground_entity.add_component(pd_comp)
        
        pd_comp.set_pose_twist(ground_pose, ground_twist)

    return ground_entity


def init_cloth(scene: sapien.Scene):
    cloth_path = os.path.join(os.path.dirname(__file__), "../assets/cloth_51.obj")
    vertices, faces = igl.read_triangle_mesh(cloth_path)
    cloth_comp = PDClothComponent(
        vertices,
        faces,
        thickness=1e-3,
        density=1e3,
        stretch_stiffness=1e3,
        bend_stiffness=1e-3,
        friction=0.1,
        collider_iterpolation_depth=0,
    )
    # print(cloth_comp.cons_ptrs[:, 1] - cloth_comp.cons_ptrs[:, 0])
    # print(cloth_comp.cons_A)
    print("total cloth mass:", cloth_comp.masses.sum())

    cloth_render = sapien.render.RenderCudaMeshComponent(len(vertices), 2 * len(faces))
    cloth_render.set_vertex_count(len(vertices))
    cloth_render.set_triangle_count(2 * len(faces))
    cloth_render.set_triangles(np.concatenate([faces, faces[:, ::-1]], axis=0))
    cloth_render.set_material(
        sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.4, 1.0])
    )

    cloth_entity = sapien.Entity()
    cloth_entity.add_component(cloth_comp)
    cloth_entity.add_component(cloth_render)
    # cloth_entity.set_pose(sapien.Pose([0, 0, 0.8], [ 0.9486833, 0, 0, 0.3162278 ]))
    cloth_entity.set_pose(sapien.Pose([0, 0, 0.5]))
    # cloth_entity.set_pose(sapien.Pose([0, 0, -0.5], [0.7071068, 0, -0.7071068, 0]))

    scene.add_entity(cloth_entity)

    # set masks manually
    masks_np = np.ones(len(vertices), dtype=np.float32)
    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    for i in range(len(vertices)):
        x, y, z = vertices[i]
        if x == x_min and (y == y_min or y == y_max):
            masks_np[i] = 0.0
    p_beg, p_end = cloth_comp.particle_ptr_in_sys
    sys: PDSystem = scene.get_system("pd")
    # wp_slice(sys.masks, p_beg, p_end).assign(masks_np)

    # TODO: investigate the problem of over friction
    # cloth covers the cow, the cow moves forward
    # then the cloth moves forward as well, but eventually falls off from the **front** of the cow

    return cloth_entity


def main(n_time_steps=10000, render_every=10, save_dir=None):
    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1, 0, -1], [1, 1, 1], True)
    # scene.set_environment_map("../assets/env.ktx")
    # scene.add_ground(0.0)

    cam = init_camera(scene)

    config = PDConfig()
    # config.collision_margin = 1e-2
    # config.collision_weight = 5e2
    # config.n_pd_iters = 20
    # config.chebyshev_flag = False
    config.n_pd_iters = 20
    system = PDSystem(config=config, device="cuda:0")
    scene.add_system(system)

    cloth_entity = init_cloth(scene)
    ground_entity = init_ground(scene)

    viewer = sapien.utils.Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_pose(cam.get_entity_pose())
    viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)
    viewer.paused = True

    sapienpd.scene_update_render(scene, set_body_pose=True)
    viewer.render()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for step in range(n_time_steps):
        system.step()
        
        print(f"Step {step}, body wrenches:")
        for body in ground_entity.get_components():
            if isinstance(body, PDBodyComponent):
                print(body.name, body.f_ext_slice.numpy())

        # NP = system.n_particles
        # v_prev = system.qd_prev_step.numpy()[:NP].mean(axis=0)
        # v = system.qd.numpy()[:NP].mean(axis=0)
        # a = (v - v_prev) / config.time_step
        # logger.info(f"Step {step}, a: {a}, |a|: {np.linalg.norm(a)}, a/|a|: {a / np.linalg.norm(a)}")

        if (step + 1) % render_every == 0:
            sapienpd.scene_update_render(scene, set_body_pose=True)
            viewer.render()

            if save_dir is not None:
                cam.take_picture()
                rgba = cam.get_picture("Color")
                rgba = np.clip(rgba, 0, 1)[:, :, :3]
                rgba = Image.fromarray((rgba * 255).astype(np.uint8))
                rgba.save(
                    os.path.join(save_dir, f"step_{(step + 1) // render_every:04d}.png")
                )


wp.init()
main(n_time_steps=1000000)
