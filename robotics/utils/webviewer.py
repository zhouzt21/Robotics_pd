import numpy as np
from typing import cast, Any, Optional
from sapien import Entity, Pose
from sapien.render import RenderBodyComponent, RenderShape, RenderShapeBox, RenderShapeTriangleMesh, RenderShapeTriangleMeshPart
import open3d as o3d


def shape2render_obj(shape: RenderShape) -> o3d.geometry.Geometry3D:
    # shape = shapes[0]
    #if isinstance(component, PhysxArticulationLinkComponent):  # articulation link
    #    pose = shape.local_pose
    #else:
    #    pose = component.entity.pose * shape.local_pose
    if isinstance(shape, RenderShapeBox):
        #collision_geom = Box(side=shape.half_size * 2)
        o3d_shape =  o3d.geometry.TriangleMesh.create_box(*shape.half_size * 2)
    elif isinstance(shape, RenderShapeTriangleMesh):
        assert len(shape.parts) == 1, "Only support single part for now"
        vertices = []
        indexes = []
        total = 0
        for k in shape.parts:
            vertices.append(k.vertices)
            indexes.append(k.triangles + total)
            total += len(k.vertices)
        o3d_shape =  o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(np.concatenate(vertices)), triangles=o3d.utility.Vector3iVector(np.concatenate(indexes)))
    else: 
        raise NotImplementedError(f"Shape {shape} not supported")
    return o3d_shape


class OurRenderShape:
    def __init__(self, shape: RenderShape, local_pose: Pose, o3d_shape: Any, entity: Optional[Entity] = None) -> None:
        self.shape = shape
        self.local_pose = local_pose
        self.o3d_shape = o3d_shape
        self.entity = entity

class CustomizedViewer:
    def __init__(self) -> None:
        self.render_shapes: dict[str, list[OurRenderShape]] = {}

    def add_entity(self, entity: Entity):
        component = entity.find_component_by_type(RenderBodyComponent)
        assert (
            component is not None
        ), f"No RenderBodhyComponent found in {entity.name}: {entity.components=}"

        component = cast(RenderBodyComponent, component)
        shapes = component.render_shapes

        if entity.name in self.render_shapes:
            assert len(shapes) == len(self.render_shapes[entity.name]), "Number of shapes mismatch"
            for shape, cur_col_shape in zip(shapes, self.render_shapes[entity.name]):
                cur_col_shape.entity = entity
                cur_col_shape.shape = shape
            return
        
        assert len(shapes) > 0, "No collision shapes found in entity"

        col_shape: list[OurRenderShape] = []
        for idx, shape in enumerate(shapes):
            shape.local_pose
            o3d_obj = shape2render_obj(shape)
            col_shape.append(OurRenderShape(shape, shape.local_pose, o3d_obj, entity))
        self.render_shapes[entity.name] = col_shape

    def update_entity(self, name: str, pose: Optional[Pose]=None):
        for shape in self.render_shapes[name]:
            pose = pose or (shape.entity.get_pose() if shape.entity else None)
            # shape.local_pose = pose
            assert pose is not None, f"Pose is None for {name}"
            pose = pose * shape.local_pose
            shape.o3d_shape.transform(pose.to_transformation_matrix())

    def update_world(self):
        for name in self.render_shapes:
            self.update_entity(name)

    def update_all(self):
        for name in self.render_shapes:
            self.update_entity(name)

    def view_all(self):
        vis = o3d.visualization.Visualizer() # type: ignore
        vis.create_window()
        for name in self.render_shapes:
            for shape in self.render_shapes[name]:
                vis.add_geometry(shape.o3d_shape)
        vis.run()
        vis.destroy_window()

            

if __name__ == "__main__":
    import sapien
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    # scene.load_widget_from_package("demo_arena", "DemoArena")

    material = renderer.create_material()
    material.base_color = [0.5, 0.5, 0.5, 1]
    scene.add_ground(-1, render_material=material)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])


    loader = scene.create_urdf_loader()
    loader.multiple_collisions_decomposition = "coacd"
    loader.load_multiple_collisions_from_file = True
    loader.fix_root_link = True
    robot = loader.load('../assets/urdf-r/urdf/urdf-r.urdf')
    robot.name = 'robot'

    
    viwer = CustomizedViewer()

    robot.set_qpos(np.zeros(robot.dof))
    for link in robot.get_links():
        viwer.add_entity(link.entity)

    viwer.update_all()
    viwer.view_all()