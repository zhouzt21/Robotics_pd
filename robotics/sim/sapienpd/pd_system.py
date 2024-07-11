import typing
from typing import Union, List

import numpy as np
import warp as wp
import sapien

from .utils.array import wp_slice
from .utils.logger import sapienpd_logger as logger
from .pd_config import PDConfig
from .pd_component import *
from .pd_defs import ShapeTypes
from .kernels.step_kernels import *


class PDSystem(sapien.System):
    def __init__(self, config=None, device="cuda:0"):
        super().__init__()

        wp.init()  # TODO

        self.name = "pd"
        self.config = config or PDConfig()
        self.device = device

        self.scenes = []
        self.components = []

        self.body_components: List[PDBodyComponent] = []

        self._init_arrays()
        self._init_counters()

    def get_name(self):
        return self.name

    def _init_arrays(self):
        MS = self.config.max_scenes
        MB = self.config.max_bodies
        MBS = self.config.max_body_shapes
        MO = self.config.max_components
        MP = self.config.max_particles
        MC = self.config.max_constraints
        MM = self.config.max_constraint_sizes_sum
        HD = self.config.hash_grid_dim
        MI = self.config.max_colliders

        with wp.ScopedDevice(self.device):
            # Component data (one value per component)
            # self.component_fric = wp.zeros(MO, dtype=wp.float32)  # moved to shape_fric and particle_fric

            # Body data
            self.body_shape_types = wp.zeros(
                MB, dtype=wp.int32
            )  # one of the members of ShapeTypes
            self.body_q = wp.zeros(MB, dtype=wp.transform)  # cm to world, (p, q)
            self.body_q_prev = wp.zeros(MB, dtype=wp.transform)  # previous time step
            self.body_qd = wp.zeros(
                MB, dtype=wp.spatial_vector
            )  # cm twist in world frame, (omega, v)
            self.body_f_ext = wp.zeros(MB, dtype=wp.spatial_vector)  
            # external wrench in world frame, (torque, force)
            self.body_component_ids = wp.zeros(
                MB, dtype=wp.int32
            )  # body id -> component id

            # Shape data (one body <-> multiple shapes)
            self.shape_scales = wp.zeros(MBS, dtype=wp.vec3)  # [x, y, z]
            self.shape_volumes = wp.zeros(MBS, dtype=wp.uint64)
            self.shape_shape2cm = wp.zeros(MBS, dtype=wp.transform)  # shape to body
            self.shape_body_ids = wp.zeros(MBS, dtype=wp.int32)  # shape id -> body id
            self.shape_fric = wp.zeros(MBS, dtype=wp.float32)

            # Particle data
            self.q = wp.zeros(MP, dtype=wp.vec3)  # position
            self.qd = wp.zeros(MP, dtype=wp.vec3)  # velocity
            self.q_prev_step = wp.zeros(MP, dtype=wp.vec3)  # previous time step
            self.qd_prev_step = wp.zeros(MP, dtype=wp.vec3)
            self.q_rest = wp.zeros(MP, dtype=wp.vec3)  # rest position
            self.masks = wp.zeros(
                MP, dtype=wp.float32
            )  # 1.0 if active, 0.0 if inactive
            self.masses = wp.zeros(MP, dtype=wp.float32)
            self.particle_component_ids = wp.zeros(
                MP, dtype=wp.int32
            )  # particle id -> component id
            self.particle_fric = wp.zeros(MP, dtype=wp.float32)

            # Collider data
            self.collider_q = wp.zeros(MI, dtype=wp.vec3)  # collider position
            self.collider_indices = wp.zeros(
                (MI, 3), dtype=wp.int32
            )  # indices in q to interpolate
            self.collider_weights = wp.zeros(
                (MI, 3), dtype=wp.float32
            )  # weights for interpolation

            # Elastic constraint data
            self.elas_weights = wp.zeros(MC, dtype=wp.float32)
            self.elas_params = wp.zeros(MC, dtype=wp.float32)
            """ Elastic constraint parameters:
                rest lengths for springs
                rest mean curvature norm for bending """
            self.elas_A_indices = wp.zeros(MM, dtype=wp.int32)  # particle ids
            self.elas_A_indptr = wp.zeros(
                (MC, 2), dtype=wp.int32
            )  # particle ids pointers (start, end)
            # I'm not directly using CSR indptr here because it might be difficult to modify (add/remove components)
            # Not sure if it's a good idea. We can change it to CSR format or warp sparse matrix later.
            self.elas_A_data = wp.zeros(
                MM, dtype=wp.float32
            )  # PD L2 penalty: |A * q - p|^2
            """ For each constraint, let M be # of particles involved, store A as a 1xM matrix
                PD L2 penalty: |(A ⊗ I_3)Sq - p|^2, where ⊗ is the Kronecker product
                A is a 1xM matrix, I3 is the 3x3 identity matrix, Sq is a 3M vector, 
                S is the matrix that selects M indices from all particles, p is a 3 vector.
                
                Since the constraints are of varying length, we concatenate all A matrices into a single array:
                A = self.elas_A_data[self.elas_A_indptr[0]:self.elas_A_indptr[1]]
                M = self.elas_A_indptr[1] - self.elas_A_indptr[0]
                Involved particle ids: self.elas_A_indices[self.elas_A_indptr[0]:self.elas_A_indptr[1]] """

            # PD data, will be used inside self.step()
            self.s = wp.zeros(MP, dtype=wp.vec3)  # kinematic update target
            # self.b = wp.zeros(P, dtype=wp.vec3)  # PD linear system: Pv = b
            self.P_diag = wp.zeros(
                MP, dtype=wp.float32
            )  # diagonal of the elastic P matrix
            self.collision_P = wp.zeros(
                MP, dtype=wp.mat33
            )  # the collision P matrix (block diagonal)
            self.qd_next_iter = wp.zeros(MP, dtype=wp.vec3)  # for chebyshev update
            self.qd_prev_iter = wp.zeros(MP, dtype=wp.vec3)  # for chebyshev update
            self.f = wp.zeros(MP, dtype=wp.vec3)
            """ v' = D^{-1} (b - R * v + r), f := b - R * v 
                Meaning: f is "force" excluding the friction term """
            self.r = wp.zeros(MP, dtype=wp.vec3)  # contact force
            self.q_prev_ccd = wp.zeros(MP, dtype=wp.vec3)  # previous CCD step
            self.ccd_step = wp.zeros(MS, dtype=wp.float32)
            self.hash_grid = wp.HashGrid(dim_x=HD, dim_y=HD, dim_z=HD)
            self.sim_step_cuda_graph = None

    def _init_counters(self):
        self.n_particles = 0
        self.n_colliders = (
            0  # collider vertices generated by interpolating the original mesh vertices
        )
        self.n_constraints = 0
        self.constraint_sizes_sum = 0
        self.n_bodies = 0
        self.n_body_shapes = 0

    def _apply_pose_to_vertices(self, pose: sapien.Pose, vertices: np.ndarray):
        T = pose.to_transformation_matrix()
        return vertices @ T[:3, :3].T + T[:3, 3]

    def _register_component_scene_get_id(self, c: sapien.Component):
        if c.entity.scene not in self.scenes:
            self.scenes.append(c.entity.scene)
        scene_id = self.scenes.index(c.entity.scene)

        component_id = len(self.components)
        self.components.append(c)

        return scene_id, component_id

    def register_body_component(self, c: PDBodyComponent):
        scene_id, c.id_in_sys = self._register_component_scene_get_id(c)
        self.body_components.append(c)

        # Component data
        c_beg, c_end = c.id_in_sys, c.id_in_sys + 1

        # Body data
        b_beg, b_end = self.n_bodies, self.n_bodies + 1
        bs_beg, bs_end = self.n_body_shapes, self.n_body_shapes + c.n_shapes
        c.body_ptr_in_sys = (b_beg, b_end)

        cm2world = c.entity.pose * c.cm2body  # pose is body to world
        wp_slice(self.body_q, b_beg, b_end).fill_(
            wp.transform(cm2world.p, np.concatenate((cm2world.q[1:], cm2world.q[:1])))
        )
        wp_slice(self.body_qd, b_beg, b_end).fill_(
            wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        wp_slice(self.body_f_ext, b_beg, b_end).fill_(0.0)
        wp_slice(self.body_component_ids, b_beg, b_end).fill_(c.id_in_sys)

        # Shape data
        wp_slice(self.body_shape_types, bs_beg, bs_end).assign(c.shape_types)
        wp_slice(self.shape_scales, bs_beg, bs_end).assign(c.scales)
        wp_slice(self.shape_fric, bs_beg, bs_end).assign(c.frictions)
        wp_slice(self.shape_volumes, bs_beg, bs_end).assign(c.volume_ids)
        wp_slice(self.shape_shape2cm, bs_beg, bs_end).assign(
            [
                wp.transform(
                    shape2cm.p, np.concatenate((shape2cm.q[1:], shape2cm.q[:1]))
                )
                for shape2cm in c.shape2cm
            ]
        )
        wp_slice(self.shape_body_ids, bs_beg, bs_end).fill_(b_beg)

        # Update counters
        self.n_bodies += 1
        self.n_body_shapes += c.n_shapes

    def register_cloth_component(self, c: PDClothComponent):
        scene_id, c.id_in_sys = self._register_component_scene_get_id(c)

        # Component data
        c_beg, c_end = c.id_in_sys, c.id_in_sys + 1

        # Particle data
        p_beg, p_end = self.n_particles, self.n_particles + c.n_vertices
        c.particle_ptr_in_sys = (p_beg, p_end)
        vertices_world = self._apply_pose_to_vertices(c.entity.pose, c.vertices)
        wp_slice(self.q, p_beg, p_end).assign(vertices_world)
        wp_slice(self.qd, p_beg, p_end).zero_()
        # q_prev_step and qd_prev_step will be assigned in self.step()
        wp_slice(self.q_rest, p_beg, p_end).assign(c.vertices)
        wp_slice(self.masks, p_beg, p_end).fill_(1.0)
        wp_slice(self.masses, p_beg, p_end).assign(c.masses)
        wp_slice(self.particle_component_ids, p_beg, p_end).fill_(c.id_in_sys)
        wp_slice(self.particle_fric, p_beg, p_end).fill_(c.friction)

        # Collider data
        col_beg, col_end = self.n_colliders, self.n_colliders + c.n_colliders
        # collider_q will be assigned in self.step()
        wp_slice(self.collider_indices, col_beg, col_end).assign(
            c.collider_indices + p_beg
        )
        wp_slice(self.collider_weights, col_beg, col_end).assign(c.collider_weights)

        # Elastic constraint data
        c_beg, c_end = self.n_constraints, self.n_constraints + c.n_constraints
        m_beg = self.constraint_sizes_sum
        m_end = m_beg + c.cons_sizes_sum
        wp_slice(self.elas_weights, c_beg, c_end).assign(c.cons_weights)
        wp_slice(self.elas_params, c_beg, c_end).assign(c.cons_params)
        wp_slice(self.elas_A_indices, m_beg, m_end).assign(c.cons_A_indices + p_beg)
        wp_slice(self.elas_A_data, m_beg, m_end).assign(c.cons_A_data)
        wp_slice(self.elas_A_indptr, c_beg, c_end).assign(c.cons_A_indptr + m_beg)

        # Update counters
        self.n_particles += c.n_vertices
        self.n_colliders += c.n_colliders
        self.n_constraints += c.n_constraints
        self.constraint_sizes_sum += c.cons_sizes_sum

    def sync_body(self):
        for b in self.body_components:
            if b.source:
                b.set_pose_twist(
                    b.entity_pose,
                    np.concatenate(
                        [b.source.angular_velocity, b.source.linear_velocity]
                    ),
                )

    def step(self):
        h = self.config.time_step
        g = self.config.gravity
        collision_margin = self.config.collision_margin
        collision_weight = self.config.collision_weight
        collision_sphere_radius = self.config.collision_sphere_radius
        n_pd_iters = self.config.n_pd_iters
        ccd_flag = self.config.ccd_flag
        ccd_every = self.config.ccd_every
        max_velocity = self.config.max_velocity
        hash_grid_radius = (
            collision_sphere_radius * 2.0 + collision_margin + max_velocity * h
        )

        if ccd_flag:
            raise NotImplementedError("CCD is not implemented yet.")

        NP = self.n_particles
        NC = self.n_constraints
        NI = self.n_colliders

        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=compute_collider_q,
                dim=NI,
                inputs=[
                    self.collider_indices,
                    self.collider_weights,
                    self.q,
                    self.collider_q,
                ],
            )
            self.hash_grid.build(
                wp_slice(self.collider_q, 0, self.n_colliders), hash_grid_radius
            )

            wp.copy(self.q_prev_step, self.q, count=NP)
            wp.copy(self.qd_prev_step, self.qd, count=NP)
            wp.copy(self.q_prev_ccd, self.q, count=NP)
            wp.copy(self.qd_prev_iter, self.qd, count=NP)

            wp.copy(self.body_q_prev, self.body_q, count=self.n_bodies)

            # wp.copy(self.P_diag, self.masses, count=NP)  # init P = M
            wp.launch(
                kernel=init_P_diag,
                dim=NP,
                inputs=[self.masses, self.config.damping, h],
                outputs=[self.P_diag],
            )
            wp.launch(
                kernel=compute_P_diag_elastic,
                dim=NC,
                inputs=[
                    self.elas_weights,
                    self.elas_params,
                    self.elas_A_indices,
                    self.elas_A_indptr,
                    self.elas_A_data,
                    h,
                    self.P_diag,
                ],
            )

            wp.launch(
                kernel=kinematic_update,
                dim=NP,
                inputs=[self.q, self.qd, self.masks, h, g, 0.0, self.s],
            )
            wp.launch(
                kernel=kinematic_update_body,
                dim=self.n_bodies,
                inputs=[
                    self.body_q,
                    self.body_qd,
                    h,
                ],
            )

            for pd_step in range(n_pd_iters):
                wp.launch(
                    kernel=init_f,
                    dim=NP,
                    inputs=[self.s, self.masks, self.masses, self.f],
                )
                wp.launch(
                    kernel=compute_f_elastic,
                    dim=NC,
                    inputs=[
                        self.q,
                        self.qd,
                        self.q_prev_step,
                        self.masks,
                        self.elas_weights,
                        self.elas_params,
                        self.elas_A_indices,
                        self.elas_A_indptr,
                        self.elas_A_data,
                        h,
                        self.f,
                    ],
                )
                self.r.zero_()
                self.collision_P.zero_()
                self.body_f_ext.zero_()
                wp.launch(
                    kernel=compute_collision_point_body,
                    dim=NI,
                    inputs=[
                        self.n_body_shapes,
                        self.collider_indices,
                        self.collider_weights,
                        self.masses,
                        self.q,
                        self.qd,
                        self.q_prev_step,
                        self.particle_fric,
                        self.body_shape_types,
                        self.shape_scales,
                        self.shape_volumes,
                        self.body_q,
                        # self.body_qd,
                        self.body_q_prev,
                        self.shape_body_ids,
                        self.shape_shape2cm,
                        self.shape_fric,
                        collision_weight,
                        collision_sphere_radius,
                        h,
                        collision_margin,
                        self.f,
                        self.P_diag,
                    ],
                    outputs=[
                        self.r,
                        self.collision_P,
                        self.body_f_ext,
                    ],
                )
                wp.launch(
                    kernel=compute_collision_point_point,
                    dim=NI,
                    inputs=[
                        self.hash_grid.id,
                        self.collider_indices,
                        self.collider_weights,
                        self.q,
                        self.qd,
                        self.q_prev_step,
                        self.qd_prev_step,
                        self.q_rest,
                        self.masses,
                        self.particle_component_ids,
                        self.particle_fric,
                        collision_weight,
                        collision_sphere_radius,
                        h,
                        collision_margin,
                        self.f,
                        self.P_diag,
                    ],
                    outputs=[
                        self.r,
                        self.collision_P,
                    ],
                )
                wp.launch(
                    kernel=jacobi_step_v,
                    dim=NP,
                    inputs=[
                        self.f,
                        self.r,
                        self.P_diag,
                        self.collision_P,
                        self.masks,
                    ],
                    outputs=[
                        self.qd_next_iter,
                    ],
                )

                if self.config.chebyshev_flag:
                    rho = self.config.chebyshev_rho
                    gamma = self.config.chebyshev_gamma
                    if pd_step < self.config.chebyshev_warmup_iters:
                        omega = 1.0
                    elif pd_step == self.config.chebyshev_warmup_iters:
                        omega = 2.0 / (2.0 - rho * rho)
                    else:
                        omega = 4.0 / (4.0 - rho * rho * omega)
                    wp.launch(
                        kernel=chebyshev_update,
                        dim=NP,
                        inputs=[
                            self.qd_prev_iter,
                            self.qd,
                            self.qd_next_iter,
                            omega,
                            gamma,
                        ],
                    )
                else:
                    wp.copy(self.qd_prev_iter, self.qd, count=NP)
                    wp.copy(self.qd, self.qd_next_iter, count=NP)

                wp.launch(
                    kernel=compute_q_from_qd,
                    dim=NP,
                    inputs=[self.q_prev_step, self.qd, h, self.q],
                )
