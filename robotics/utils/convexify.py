import os
import trimesh
import argparse
from typing import Union, Optional
from sapien.wrapper.coacd import do_coacd #TODO: support coacd


from robotics.utils.path import PACKAGE_ASSET_DIR

def as_mesh(scene_or_mesh: Union[trimesh.Scene, trimesh.Trimesh, str]) -> Optional[trimesh.Trimesh]:
    if isinstance(scene_or_mesh, str):
        scene_or_mesh = trimesh.load_mesh(scene_or_mesh) # type: ignore
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh

    assert mesh is None or isinstance(mesh, trimesh.Trimesh)
    return mesh



def run_vhacd(mesh: trimesh.Trimesh):
    vhacd = os.path.join(PACKAGE_ASSET_DIR, 'v-hacd')
    if not os.path.exists(vhacd):
        os.system(f"git clone https://github.com/kmammou/v-hacd.git {vhacd}")
        os.system(f"cd {vhacd}/app && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build")


    mesh.export("/tmp/tmp.obj")
    os.system(f"cd /tmp; {vhacd}/app/build/TestVHACD /tmp/tmp.obj -o obj")
    return trimesh.load_mesh("/tmp/decomp.stl")


def convexify(mesh_or_path: Union[trimesh.Trimesh, str], output_path: Optional[str], method='vhacd', skip_if_exists=True):
    if output_path is not None and os.path.exists(output_path) and skip_if_exists:
        return trimesh.load_mesh(output_path)
    mesh = as_mesh(mesh_or_path)
    if mesh is None:
        raise ValueError(f'Empty mesh {mesh_or_path}')

    if method == 'vhacd':
        mesh = run_vhacd(mesh)
    else:
        raise NotImplementedError(f'Unknown method {method}')

    if mesh is None:
        raise ValueError(f'mehs convixfy failed {mesh_or_path}')

    assert isinstance(mesh, trimesh.Trimesh) or isinstance(mesh, trimesh.Scene)
    if output_path is not None:
        assert output_path.endswith('.obj'), "sapien doesn't support stl for multiple convex shapes"
        mesh.export(output_path)
    return mesh



#TODO: convexify urdf
# logger = logging.getLogger("pymp.robot")
# def cook_urdf_for_pinocchio(urdf_path, use_convex, package_dir: Optional[str] = None):
#     with open(urdf_path, "r") as f:
#         urdf_xml = BeautifulSoup(f.read(), "xml")

#     # for mesh_tag in urdf_xml.find_all("mesh"):
#     for mesh_tag in urdf_xml.select("collision mesh"):
#         filename = mesh_tag["filename"].lstrip("package://")

#         # Use convex collision shape
#         if use_convex:
#             assert (
#                 package_dir is not None
#             ), "package_dir must be specified if using convex collision"
#             # First check if the (SAPIEN) convex hull file exists
#             convex_path = os.path.join(package_dir, filename + ".convex.stl")
#             if os.path.exists(convex_path):
#                 filename = '/' + filename + ".convex.stl"
#             else:
#                 # Then check if the (trimesh) convex hull file exists
#                 convex2_path = os.path.join(package_dir, filename + ".convex.stl")
#                 if not os.path.exists(convex2_path):
#                     logger.info(
#                         "Convex hull ({}) not found, generating...".format(convex2_path)
#                     )
#                     mesh_path = os.path.join(package_dir, filename)

#                     mesh = as_mesh(trimesh.load_mesh(mesh_path))
#                     cvx_mesh = trimesh.convex.convex_hull(mesh)
#                     cvx_mesh.export(convex2_path)
#                 filename = filename + ".convex2.stl"

#         # Add "package://" for Pinocchio
#         # mesh_tag["filename"] = "package://" + filename
#         mesh_tag["filename"] = filename

#     return urdf_xml


def main():
    parser = argparse.ArgumentParser(description='Convert DAE to OBJ')
    parser.add_argument('--input', type=str, help='Input DAE file')
    parser.add_argument('--output', type=str, help='Output OBJ file')
    args = parser.parse_args()
    raise NotImplementedError


    
if __name__ == '__main__':
    main()