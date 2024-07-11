import os
import numpy as np
import igl
from PIL import Image
import warp as wp
import sapien

import sapienpd
from sapienpd.pd_config import PDConfig
from sapienpd.pd_system import PDSystem
from sapienpd.pd_component import PDPlaneComponent, PDClothComponent
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
        sapien.Pose([-0.064028, -0.270365, 0.440705], [0.73446, -0.146693, 0.1681, 0.64093])
    )
    scene.add_entity(cam_entity)
    return cam


def init_ground(scene: sapien.Scene):
    ground_comp = PDPlaneComponent(
        normal=np.array([1, 0, 0], dtype=np.float32), 
        offset=0.0, 
        friction=0.1
    )
    ground_render = sapien.render.RenderBodyComponent()
    ground_render.attach(
        sapien.render.RenderShapePlane(
            np.array([10., 10., 10.]),
            sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0])
        )
    )
    ground_entity = sapien.Entity()
    ground_entity.add_component(ground_comp)
    ground_entity.add_component(ground_render)
    ground_entity.set_pose(sapien.Pose(p=[0, 0, 0], q=[ 0.7071068, 0, -0.7071068, 0 ]))
    scene.add_entity(ground_entity)

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
    # print(cloth_comp.masses, cloth_comp.masses.sum())

    cloth_render = sapien.render.RenderCudaMeshComponent(len(vertices), 2 * len(faces))
    cloth_render.set_vertex_count(len(vertices))
    cloth_render.set_triangle_count(2 * len(faces))
    cloth_render.set_triangles(np.concatenate([faces, faces[:, ::-1]], axis=0))
    cloth_render.set_material(sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.4, 1.0]))

    cloth_entity = sapien.Entity()
    cloth_entity.add_component(cloth_comp)
    cloth_entity.add_component(cloth_render)
    cloth_entity.set_pose(sapien.Pose([0, 0, 0.5]))

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
    wp_slice(sys.masks, p_beg, p_end).assign(masks_np)

    return cloth_entity


def main(n_time_steps=10000, render_every=10, save_dir=None):
    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([1, 0, -1], [1, 1, 1], True)
    # scene.add_ground(0.0)

    cam = init_camera(scene)

    config = PDConfig()
    # config.n_pd_iters = 20
    # config.chebyshev_flag = False
    system = PDSystem(config=config, device="cuda:0")
    scene.add_system(system)

    cloth_entity = init_cloth(scene)
    ground_entity = init_ground(scene)

    viewer = sapien.utils.Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_pose(cam.get_entity_pose())
    viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)
    viewer.paused = True

    sapienpd.scene_update_render(scene)
    viewer.render()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for step in range(n_time_steps):
        system.step()
        if (step + 1) % render_every == 0:
            sapienpd.scene_update_render(scene)
            viewer.render()

            if save_dir is not None:
                cam.take_picture()
                rgba = cam.get_picture("Color")
                rgba = np.clip(rgba, 0, 1)[:, :, :3]
                rgba = Image.fromarray((rgba * 255).astype(np.uint8))
                rgba.save(os.path.join(save_dir, f"step_{(step + 1) // render_every:04d}.png"))


wp.init()
main(n_time_steps=2000000)
