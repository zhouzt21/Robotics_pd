import trimesh
from typing import Dict, List, TypeVar, Union, Any, Type, overload

from sapien import physx
from copy import deepcopy

import numpy as np
import sapien
import sapien.render
import sapien.wrapper.urdf_loader
import sapien.physx as physx
#from sapien import Pose
from robotics import Pose


T = TypeVar("T")


def get_component_meshes(component: physx.PhysxRigidBaseComponent) -> trimesh.Trimesh:
    """Get component (collision) meshes in the component's frame."""
    meshes = []
    for geom in component.get_collision_shapes():
        if isinstance(geom, physx.PhysxCollisionShapeBox):
            mesh = trimesh.creation.box(extents=2 * geom.half_size)
        elif isinstance(geom, physx.PhysxCollisionShapeCapsule):
            mesh = trimesh.creation.capsule(
                height=2 * geom.half_length, radius=geom.radius
            )
        elif isinstance(geom, physx.PhysxCollisionShapeSphere):
            mesh = trimesh.creation.icosphere(radius=geom.radius)
        elif isinstance(geom, physx.PhysxCollisionShapeCylinder):
            mesh = trimesh.creation.cylinder(radius=geom.radius, height=geom.half_length * 2)
        elif isinstance(geom, physx.PhysxCollisionShapePlane):
            continue
        elif isinstance(
            geom, (physx.PhysxCollisionShapeConvexMesh)
        ):
            vertices = geom.vertices  # [n, 3]
            faces = geom.get_triangles()
            vertices = vertices * geom.scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        elif isinstance(geom, physx.PhysxCollisionShapeTriangleMesh):
            assert (geom.scale == 1).all(), f"scale not supported {geom.scale}"
            mesh = trimesh.Trimesh(vertices=geom.vertices, faces=geom.triangles)
        else:
            raise TypeError(type(geom))
        mesh.apply_transform(geom.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_obj_by_name(objs: List[T], name: str, is_unique=True):
    """Get a objects given the name.

    Args:
        objs (List[T]): objs to query. Expect these objects to have a get_name function. These may be sapien.Entity, physx.PhysxArticulationLink etc.
        name (str): name for query.
        is_unique (bool, optional):
            whether the name should be unique. Defaults to True.

    Raises:
        RuntimeError: The name is not unique when @is_unique is True.

    Returns:
        T or List[T]:
            matched T or Ts. None if no matches.
    """
    matched_objects = [x for x in objs if x.get_name() == name]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same name {name}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None



from typing import Dict, List
## TODO clean up the code here, too many functions that are plurals of one or the other and confusing naming
import numpy as np
import sapien
import sapien.render
import trimesh
import trimesh.creation
import sapien.physx as physx

def get_render_body_component(entity: sapien.Entity) -> sapien.render.RenderBodyComponent:
    """Get sapien.render.RenderBodyComponent. Assumes entity only ever has one of these 

    Returns: sapien.renderRenderBodyComponent if it exists, None otherwise
    """
    return get_obj_by_type(entity.components, sapien.render.RenderBodyComponent)


def get_visual_body_meshes(visual_body: sapien.render.RenderBodyComponent):
    # TODO (stao): update this
    meshes = []
    for render_shape in visual_body.render_shapes:
        # TODO (stao): do other render shapes permit changing material?
        # TODO (stao): can there be multiple renderbody components, or is it unique?
        if type(render_shape) == sapien.render.RenderShapeTriangleMesh:
            for part in render_shape.parts:
                vertices = part.vertices * render_shape.scale  # [n, 3]
                faces = part.triangles
                # faces = render_shape.mesh.indices.reshape(-1, 3)  # [m * 3]
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.apply_transform(render_shape.local_pose.to_transformation_matrix())
                meshes.append(mesh)
    return meshes


def get_actor_visual_meshes(actor: sapien.Entity):
    """Get actor (visual) meshes in the actor frame."""
    meshes = []
    comp = get_render_body_component(actor)
    if comp is not None:
        meshes.extend(get_visual_body_meshes(comp))
    return meshes




def get_component_mesh(component: physx.PhysxRigidBaseComponent, to_world_frame=True):
    mesh = merge_meshes(get_component_meshes(component))
    if mesh is None:
        return None
    if to_world_frame:
        T = component.entity_pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_actor_visual_mesh(actor: sapien.Entity):
    mesh = merge_meshes(get_actor_visual_meshes(actor))
    if mesh is None:
        return None
    return mesh


def get_articulation_meshes(
    articulation: physx.PhysxArticulation, exclude_link_names=()
):
    """Get link meshes in the world frame."""
    meshes = []
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        mesh = get_component_mesh(link, True)
        if mesh is None:
            continue
        meshes.append(mesh)
    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def get_actor_mesh(actor: sapien.Entity, transform=True):
    meshes = [get_component_meshes(comp) for comp in actor.get_components() if isinstance(comp, physx.PhysxBaseComponent)]
    meshes = [m for m in meshes if m is not None]
    mesh = merge_meshes(sum(meshes, []))
    if transform and mesh is not None:
        mesh.apply_transform(actor.get_pose().to_transformation_matrix())
    return mesh



def parse_urdf_config(config_dict: dict, scene: sapien.Scene) -> Dict:
    """Parse config from dict for SAPIEN URDF loader.

    Args:
        config_dict (dict): a dict containing link physical properties.
        scene (sapien.Scene): simualtion scene

    Returns:
        Dict: urdf config passed to `sapien.URDFLoader.load`.
    """
    urdf_config = deepcopy(config_dict)

    # Create the global physical material for all links
    mtl_cfg = urdf_config.pop("material", None)
    if mtl_cfg is not None:
        urdf_config["material"] = scene.create_physical_material(**mtl_cfg)

    # Create link-specific physical materials
    materials = {}
    for k, v in urdf_config.pop("_materials", {}).items():
        materials[k] = scene.create_physical_material(**v)

    # Specify properties for links
    for link_config in urdf_config.get("link", {}).values():
        # Substitute with actual material
        link_config["material"] = materials[link_config["material"]]

    return urdf_config



def check_urdf_config(urdf_config: dict):
    """Check whether the urdf config is valid for SAPIEN.

    Args:
        urdf_config (dict): dict passed to `sapien.URDFLoader.load`.
    """
    allowed_keys = ["material", "density", "link"]
    for k in urdf_config.keys():
        if k not in allowed_keys:
            raise KeyError(
                f"Not allowed key ({k}) for `sapien.URDFLoader.load`. Allowed keys are f{allowed_keys}"
            )

    allowed_keys = ["material", "density", "patch_radius", "min_patch_radius"]
    for k, v in urdf_config.get("link", {}).items():
        for kk in v.keys():
            # In fact, it should support specifying collision-shape-level materials.
            if kk not in allowed_keys:
                raise KeyError(
                    f"Not allowed key ({kk}) for `sapien.URDFLoader.load`. Allowed keys are f{allowed_keys}"
                )



def apply_urdf_config(loader: sapien.wrapper.urdf_loader.URDFLoader, urdf_config: dict):
    # TODO (stao): @fxiang is this complete?
    if "link" in urdf_config:
        for name, link_cfg in urdf_config["link"].items():
            if "material" in link_cfg:
                mat: physx.PhysxMaterial = link_cfg["material"]
                loader.set_link_material(name, mat.static_friction, mat.dynamic_friction, mat.restitution)
            if "patch_radius" in link_cfg:
                loader.set_link_patch_radius(name, link_cfg["patch_radius"])
            if "min_patch_radius" in link_cfg:
                loader.set_link_min_patch_radius(name, link_cfg["min_patch_radius"])
            if "density" in link_cfg:
                loader.set_link_density(name, link_cfg["density"])
            # TODO (stao): throw error if there is a config not used?
    if "material" in urdf_config:
        mat: physx.PhysxMaterial = urdf_config["material"]
        loader.set_material(mat.static_friction, mat.dynamic_friction, mat.restitution)
    if "patch_radius" in urdf_config:
        loader.set_patch_radius(urdf_config["patch_radius"])
    if "min_patch_radius" in urdf_config:
        loader.set_min_patch_radius(urdf_config["min_patch_radius"])
    if "density" in urdf_config:
        loader.set_density(urdf_config["density"])
   

def get_obj_by_type(objs: List[Any], target_type: Type[T], is_unique=True) -> Union[T, List[T], None]:
    matched_objects = [x for x in objs if type(x) == target_type]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same type {target_type}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None

def ensure_obj_by_type(objs: List[Any], target_type: Type[T]) -> T:
    output = get_obj_by_type(objs, target_type)
    assert output is not None and not isinstance(output, list), f"Cannot find object of type {target_type}"
    return output

def get_rigid_dynamic_component(entity: sapien.Entity) -> Union[physx.PhysxRigidDynamicComponent, None]:
    """Get physx.PhysxRigidDynamicComponent. Assumes entity only ever has one of these 

    Returns: physx.PhysxRigidDynamicComponent if it exists, None otherwise
    """
    output = get_obj_by_type(entity.components, physx.PhysxRigidDynamicComponent) 
    assert not isinstance(output, list)
    return output

def get_rigid_static_component(entity: sapien.Entity) -> physx.PhysxRigidStaticComponent:
    """Get physx.PhysxRigidStaticComponent. Assumes entity only ever has one of these 

    Returns: physx.PhysxRigidStaticComponent if it exists, None otherwise
    """
    return ensure_obj_by_type(entity.components, physx.PhysxRigidStaticComponent)
    

def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    else:
        return x / norm


def look_at(eye, target, up=(0, 0, 1)) -> sapien.Pose:
    """Get the camera pose in SAPIEN by the Look-At method.

    Note:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        The SAPIEN camera follows the convention: (forward, right, up) = (x, -y, z)
        while the OpenGL camera follows (forward, right, up) = (-z, x, y)
        Note that the camera coordinate system (OpenGL) is left-hand.

    Args:
        eye: camera location
        target: looking-at location
        up: a general direction of "up" from the camera.

    Returns:
        sapien.Pose: camera pose
    """
    forward = normalize_vector(np.array(target) - np.array(eye))
    up = normalize_vector(up)
    left = np.cross(up, forward)
    up = np.cross(forward, left)
    rotation = np.stack([forward, left, up], axis=1)
    import transforms3d
    mat2quat = transforms3d.quaternions.mat2quat
    return sapien.Pose(p=eye, q=mat2quat(rotation))


def hide_entity(entity: sapien.Entity):
    get_render_body_component(entity).disable()