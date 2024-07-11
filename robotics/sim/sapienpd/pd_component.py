import typing
from typing import TYPE_CHECKING, List, Union

import igl
import numpy as np
import sapien
import scipy.sparse as sp
import warp as wp
from mesh2nvdb import mesh2nvdb
from sapien.render import RenderCudaMeshComponent

from .pd_defs import ShapeTypes
from .utils.array import wp_slice
from .utils.logger import sapienpd_logger as logger
from .utils.render import copy_positions_to_render

if TYPE_CHECKING:
    from .pd_system import PDSystem


class PDBodyComponent(sapien.Component):
    @staticmethod
    def from_physx_shape(body: sapien.physx.PhysxRigidBodyComponent, grid_size=0.01):
        cm2body = body.cmass_local_pose
        shape_types = []
        scales = []
        frictions = []
        volumes = []
        shape2cm_list = []

        for shape in body.collision_shapes:
            if isinstance(shape, sapien.physx.PhysxCollisionShapeConvexMesh):
                shape_types.append(ShapeTypes.GEO_SDF)
                shape: sapien.physx.PhysxCollisionShapeConvexMesh
                nvdb = mesh2nvdb(shape.vertices, shape.triangles, grid_size)
                volume = wp.Volume.load_from_nvdb(nvdb)
                volumes.append(volume)
                assert shape.scale[0] == shape.scale[1] == shape.scale[2]
                scales.append(shape.scale)
            elif isinstance(shape, sapien.physx.PhysxCollisionShapeBox):
                shape_types.append(ShapeTypes.GEO_BOX)
                scales.append(shape.half_size)
                volumes.append(None)
            elif isinstance(shape, sapien.physx.PhysxCollisionShapeSphere):
                shape_types.append(ShapeTypes.GEO_SPHERE)
                scales.append([shape.radius] * 3)
                volumes.append(None)
            else:
                raise Exception("no implemented shape type")

            shape2body = shape.local_pose
            shape2cm_list.append(cm2body.inv() * shape2body)
            frictions.append(shape.physical_material.dynamic_friction)

        return PDBodyComponent(
            shape_types, scales, frictions, volumes, cm2body, shape2cm_list, body
        )

    def __init__(
        self,
        shape_types: List[int],
        scales: Union[np.ndarray, List[float]] = None,
        frictions: Union[np.ndarray, List[float]] = None,
        volumes: List[wp.Volume] = None,
        cm2body: sapien.Pose = None,
        shape2cm: List[sapien.Pose] = None,
        source: sapien.physx.PhysxRigidBodyComponent = None,
    ):
        super().__init__()
        self.source = source

        self.id_in_sys = None  # component id
        self.body_ptr_in_sys = None
        self.q_slice = None
        self.qd_slice = None  # (omega, v) in the world frame
        self.f_ext_slice = None  # (tau, f) in the world frame

        self.n_shapes = len(shape_types)
        self.shape_types = shape_types
        self.scales = scales if scales is not None else np.ones((self.n_shapes, 3))
        self.frictions = frictions if frictions is not None else np.zeros(self.n_shapes)
        self.volumes = volumes
        self.volume_ids = (
            [v.id if v else 0 for v in volumes]
            if volumes is not None
            else [0] * self.n_shapes
        )
        self.cm2body = cm2body if cm2body is not None else sapien.Pose()
        self.shape2cm = (
            shape2cm
            if shape2cm is not None
            else [sapien.Pose() for _ in range(self.n_shapes)]
        )
        assert (
            len(self.scales) == self.n_shapes
        ), f"scales length mismatch: got {self.n_shapes} shape_types but {len(self.scales)} scales"
        assert (
            len(self.frictions) == self.n_shapes
        ), f"frictions length mismatch: got {self.n_shapes} shape_types but {len(self.frictions)} frictions"
        assert (
            len(self.volume_ids) == self.n_shapes
        ), f"volume_ids length mismatch: got {self.n_shapes} shape_types but {len(self.volume_ids)} volume_ids"
        assert (
            len(self.shape2cm) == self.n_shapes
        ), f"shape2cm length mismatch: got {self.n_shapes} shape_types but {len(self.shape2cm)} shape2cm"

    def on_add_to_scene(self, scene: sapien.Scene):
        s: PDSystem = scene.get_system("pd")
        s.register_body_component(self)
        b_beg, b_end = self.body_ptr_in_sys
        self.q_slice = wp_slice(s.body_q, b_beg, b_end)
        self.qd_slice = wp_slice(s.body_qd, b_beg, b_end)
        self.f_ext_slice = wp_slice(s.body_f_ext, b_beg, b_end)

    def set_pose_twist(self, pose: sapien.Pose, twist: np.ndarray):
        """pose: pose of the body in the world frame, i.e. body2world
        twist: [omega_x, omega_y, omega_z, v_x, v_y, v_z]
            of the CM frame in the world frame"""
        assert len(twist) == 6
        assert self.entity is not None and self.entity.scene is not None

        if isinstance(twist, np.ndarray):
            twist = twist.astype(np.float32)

        cm2world = pose * self.cm2body
        # logger.debug(f"cm2world: {cm2world}")
        cm2world_wp = wp.transform(
            cm2world.p, np.concatenate((cm2world.q[1:], cm2world.q[:1]))
        )
        twist_wp = wp.spatial_vector(twist)
        self.q_slice.fill_(cm2world_wp)
        self.qd_slice.fill_(twist_wp)

    def update_entity_pose(self):
        assert self.entity is not None and self.entity.scene is not None

        b_i = self.body_ptr_in_sys[0]
        X_wb = self.q_slice.numpy()[b_i]
        p = X_wb[:3]
        q = X_wb[3:]
        q = np.concatenate([q[3:], q[:3]])
        cm2body = self.cm2body
        # (p, q) is cm2world
        # we need body2world = cm2world * body2cm
        self.entity.set_pose(sapien.Pose(p, q) * cm2body.inv())


class PDClothComponent(sapien.Component):
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        thickness: float,
        density: float,
        stretch_stiffness: float,
        bend_stiffness: float,
        friction: float,
        collider_iterpolation_depth: int = 1,
    ):
        """
        :param stretch_stiffness: Stretch force per unit length; unit: N/m
        :param bend_stiffness: Bending force per unit mean curvature per unit area; unit: N/m
        """

        super().__init__()

        # Properties to be set by the system
        self.id_in_sys = None
        self.particle_ptr_in_sys = None
        """ For every particle data arrays in the system,
            the slice [particle_ptr_in_sys[0], particle_ptr_in_sys[1]]
            is used by this component. 
            particle_ptr_in_sys[1] - particle_ptr_in_sys[0] = n_vertices """

        # Physics properties
        self.thickness = thickness  # only used for mass calculation
        self.density = density
        self.stretch_stiffness = stretch_stiffness  # springs
        self.bend_stiffness = bend_stiffness  # laplacian
        self.friction = friction

        # Mesh data
        assert faces.shape[1] == 3
        self.vertices = vertices.astype(np.float32)
        self.faces = faces.astype(np.int32)
        self.edges: np.ndarray = igl.edges(self.faces)

        # Derived properties
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)
        self.n_edges = len(self.edges)
        n_spring_cons = self.n_edges
        n_bending_cons = self.n_vertices
        spring_cons_sizes_sum = n_spring_cons * 2
        bending_cons_sizes_sum = self.n_edges * 2 + self.n_vertices
        self.n_constraints = n_spring_cons + n_bending_cons
        self.cons_sizes_sum = spring_cons_sizes_sum + bending_cons_sizes_sum

        # Derived properties: colliders
        self._interpolate_colliders(collider_iterpolation_depth)

        # Derived properties: Voronoi areas and mean curvature normals
        # Prepare for bending constraints and masses
        cotmatrix = igl.cotmatrix(self.vertices, self.faces)  # [N, N]
        massmatrix = igl.massmatrix(
            self.vertices, self.faces, igl.MASSMATRIX_TYPE_VORONOI
        )
        voronoi_areas = massmatrix.diagonal()
        A_bend_sp: sp.csr_matrix = -(
            cotmatrix / voronoi_areas[:, None]
        ).tocsr()  # [N, N]
        rest_mean_curvatures = A_bend_sp.dot(vertices)  # [N, 3]
        rest_mean_curvatures_norms = np.linalg.norm(rest_mean_curvatures, axis=1)  # [N]
        assert A_bend_sp.nnz == bending_cons_sizes_sum

        # Derived properties: masses
        self.masses = voronoi_areas * self.density * self.thickness

        # Derived properties: constraints
        self.cons_weights = np.zeros(self.n_constraints, dtype=np.float32)
        self.cons_weights[: self.n_edges] = self.stretch_stiffness
        self.cons_weights[self.n_edges :] = self.bend_stiffness * voronoi_areas

        # Rest lengths (for springs) and mean curvature norms (for bending)
        self.cons_params = np.zeros(self.n_constraints, dtype=np.float32)
        # Spring rest lengths
        self.cons_params[: self.n_edges] = np.linalg.norm(
            self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]], axis=1
        )
        # Bending rest mean curvature norms
        self.cons_params[self.n_edges :] = rest_mean_curvatures_norms

        # Involved particle ids
        self.cons_A_indices = np.zeros(self.cons_sizes_sum, dtype=np.int32)
        # Spring constraints
        self.cons_A_indices[:spring_cons_sizes_sum] = self.edges.reshape(-1)
        # Bending constraints (indices of A_bend_sp)
        self.cons_A_indices[spring_cons_sizes_sum:] = A_bend_sp.indices

        self.cons_A_data = np.zeros(self.cons_sizes_sum, dtype=np.float32)
        # Spring constraints
        self.cons_A_data[:spring_cons_sizes_sum] = np.tile(
            np.array([1, -1], dtype=np.float32), n_spring_cons
        )
        # Bending constraints
        self.cons_A_data[spring_cons_sizes_sum:] = A_bend_sp.data

        # Begin and end pointers to the involved particle ids array
        self.cons_A_indptr = np.zeros((self.n_constraints, 2), dtype=np.int32)
        # Spring constraints
        self.cons_A_indptr[:n_spring_cons, 0] = np.arange(0, self.n_edges * 2, 2)
        self.cons_A_indptr[:n_spring_cons, 1] = (
            self.cons_A_indptr[:n_spring_cons, 0] + 2
        )
        # Bending constraints (offsets of A_bend_sp)
        self.cons_A_indptr[self.n_edges :, 0] = (
            A_bend_sp.indptr[:-1] + spring_cons_sizes_sum
        )
        self.cons_A_indptr[self.n_edges :, 1] = (
            A_bend_sp.indptr[1:] + spring_cons_sizes_sum
        )

    def _interpolate_colliders(self, depth: int):
        """Interpolate the vertices on each triangular face to get the collider vertices.
        The original vertices are also used as colliders.
        Recursively interpolate the vertices for collider_iterpolation_depth times."""
        collider_vertices = self.vertices.copy()
        self.collider_indices = np.arange(len(self.vertices))[:, None].repeat(
            3, axis=1
        )  # [N, 3]
        self.collider_weights = np.array([[1.0, 0.0, 0.0]], dtype=np.float32).repeat(
            len(self.vertices), axis=0
        )  # [N, 3]
        self.n_colliders = len(collider_vertices)
        last_faces = self.faces.copy()

        for dep in range(depth):
            new_vertices = np.mean(collider_vertices[last_faces], axis=1)  # [LF, 3]
            # Subdivide each face in the last_faces
            # First get the edges
            new_edges = last_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(
                -1, 2
            )  # [LF*3, 2]
            # Then append the new vertices
            new_indices_x3 = (
                np.repeat(np.arange(len(new_vertices)), 3, axis=0)[:, None]
                + self.n_colliders
            )  # [LF*3, 1]

            # Compute new interpolation indices
            if dep == 0:
                # Interpolate from the original vertices
                new_collider_inter_indices = last_faces
                new_collider_inter_weights = (
                    np.array([[1.0, 1.0, 1.0]], dtype=np.float32) / 3.0
                ).repeat(len(last_faces), axis=0)
            else:
                # Interpolate from the last interpolation
                new_collider_inter_indices = self.collider_indices[
                    last_faces[:, 2]
                ]  # [LF, 3], the last vertex cannot be an original vertex
                new_collider_inter_weights = np.zeros(
                    (len(last_faces), 3), dtype=np.float32
                )
                for i in range(3):
                    inter_indices_i = new_collider_inter_indices[:, i]  # [LF]
                    for j in range(3):
                        last_faces_j = last_faces[:, j]  # [LF]
                        mask = (
                            self.collider_indices[last_faces_j]
                            == inter_indices_i[:, None]
                        )  # [LF, 3]
                        weights = np.sum(
                            self.collider_weights[last_faces_j] * mask, axis=1
                        )  # [LF]
                        new_collider_inter_weights[:, i] += weights / 3.0

            # Append the new interpolation indices
            self.collider_indices = np.vstack(
                [self.collider_indices, new_collider_inter_indices]
            )
            self.collider_weights = np.vstack(
                [self.collider_weights, new_collider_inter_weights]
            )  # [N, 3]

            last_faces = np.hstack([new_edges, new_indices_x3]).reshape(
                -1, 3
            )  # [LF*3, 3]
            collider_vertices = np.vstack([collider_vertices, new_vertices])
            self.n_colliders += len(new_vertices)

    def on_add_to_scene(self, scene: sapien.Scene):
        s: PDSystem = scene.get_system("pd")
        s.register_cloth_component(self)

    def update_render(self, render_component: RenderCudaMeshComponent):
        s: PDSystem = self.entity.scene.get_system("pd")
        with wp.ScopedDevice(s.device):
            interface = render_component.cuda_vertices.__cuda_array_interface__
            dst = wp.array(
                ptr=interface["data"][0],
                dtype=wp.float32,
                shape=interface["shape"],
                strides=interface["strides"],
                owner=False,
            )
            src = wp_slice(
                s.q,
                self.particle_ptr_in_sys[0],
                self.particle_ptr_in_sys[1],
            )
            wp.launch(
                kernel=copy_positions_to_render,
                dim=src.shape[0],
                inputs=[dst, src],
            )
            render_component.notify_vertex_updated(wp.get_stream().cuda_stream)
