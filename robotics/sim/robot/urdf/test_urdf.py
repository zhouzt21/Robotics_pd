from robotics.sim.robot.urdf import URDFTool
from robotics.utils.path import PACKAGE_ASSET_DIR
import sapien
from sapien.utils.viewer import Viewer
from sapien.wrapper.urdf_loader import URDFLoader
from sapien.physx import PhysxArticulation
from robotics import Pose



def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    #material = scene.create_physical_material(0.5, 0.5, 0.1)
    material = renderer.create_material()
    material.base_color = [0.5, 0.5, 0.5, 1]
    scene.add_ground(0, render_material=material)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1) # type: ignore
    viewer.set_camera_rpy(r=0, p=-0.3, y=0) # type: ignore


    urdf_path = f"{PACKAGE_ASSET_DIR}/mobile_sapien/mobile_manipulator_xy.urdf"
    urdf = URDFTool.from_path(urdf_path)
    urdf.remove('xarm7_base_joint')

    base_urdf = urdf.prune_from('base_link')
    _, name = base_urdf.load(scene, fix_root_link=True, pose=Pose([0, 0, 0], [1, 0, 0, 0]))
    print(name)

    arm_urdf = urdf.prune_from('xarm7_base_link')
    _, name = arm_urdf.load(scene, fix_root_link=True, pose=Pose([0, 1, 0], [1, 0, 0, 0]))
    print(name)

    # xyz="0.15 0.00023421 0.02"
    #  rpy="0 0 -1.5708"
    import transforms3d
    q = transforms3d.euler.euler2quat(0, 0, -1.5708)
    p = [0.15, 0.00023421, 0.02]
    merged = base_urdf.add(arm_urdf, Pose(p, q), 'base_link', 'xarm7_base_joint')

    merged.load(scene, fix_root_link=True, pose=Pose([0, 2, 0], [1, 0, 0, 0]), filename='merged.urdf')



    while not viewer.closed:
        scene.update_render()
        viewer.render()



if __name__ == "__main__":
    main()