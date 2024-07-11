import warp as wp
import warp.sparse as ws
import warp.optim.linear as wol

import sapien

import numpy as np
import scipy.sparse as sp
import igl
import os
from PIL import Image
import time

import keyboard


wp.init()
device = wp.get_preferred_device()

# ffmpeg -framerate 50 -i output/cloth/frames_chebyshev/step_%04d.png -vf "fps=50,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" output/cloth/chebyshev.gif

######## Hyperparameters ########
time_step = 0.002
n_render_steps = None
load_state = False
state_path = "output/cloth/states/step_0160.npy"
render_flag = True
render_vertices_flag = True
render_every = 10
save_render = False
save_state = False
n_pd_iters = 20
ccd_every = 20
use_cuda_graph = False
use_hash_grid = True
use_chebyshev = True
ccd_flag = False
compute_error_flag = False and (not use_cuda_graph)
check_convergence_flag = True and (not use_cuda_graph)
debug_nan_flag = True and (not use_cuda_graph)
debug_ccd_flag = True and (not use_cuda_graph)
enable_pp_collision = False
chebyshev_n_warmup_steps = 10  # S in the PD-Chebyshev paper
pd_rho = 0.99
pd_gamma = 0.7
gravity = wp.vec3(0, 0, -9.8)
spring_stiffness = 1e3
bending_stiffness = 1e-3
collision_weight = 5e3
velocity_damping = 0.0
enable_static_friction = True
ground_friction_mu = 0.0
cloth_friction_mu = 1.0
gripper_friction_mu = 1.0
gripper_shape = "box"
sor_factor = 1.0
ground_friction_damping = 0
cloth_friction_damping = 0
cloth_thickness = 1e-3
cloth_density = 1e3
collision_sphere_radius = 8e-3
d_hat = 1e-3
ccd_slackness = 0.9
ccd_thickness = 1e-5
hash_grid_dim = 128
max_velocity = 1.0
hash_grid_size_scale = (
    max_velocity * time_step + d_hat + 2 * collision_sphere_radius + ccd_thickness
)  # TODO: add velocity clamping / CFL condition
init_height = 0.5

######## Load mesh and preprocess ########
assets_dir = os.path.join(os.path.dirname(__file__), "../../assets")
vertices, faces = igl.read_triangle_mesh(os.path.join(assets_dir, "cloth_51.obj"))
# vertices, faces = igl.read_triangle_mesh("../../assets/shirt_complex.obj")
print(f"vertices: {vertices.shape}, faces: {faces.shape}")
vertices = vertices.astype(np.float32)
faces = faces.astype(np.int32)
edges = igl.edges(faces)

# initial transformation
# T = sapien.Pose(p=[0, 0, 0.5], q=[ 0.7660444, 0, 0.6427876, 0 ]).to_transformation_matrix()
# vertices = vertices @ T[:3, :3].T + T[:3, 3]

boundary_edges = igl.boundary_facets(faces)
boundary_vertices = np.unique(boundary_edges)
is_boundary = np.zeros(vertices.shape[0], dtype=np.int32)
is_boundary[boundary_vertices] = True

n_particles = vertices.shape[0]

# Compute cotangent Laplacian
cotmatrix = igl.cotmatrix(vertices, faces)
massmatrix = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI)
voronoi_areas = massmatrix.diagonal()
# print(f"voronoi_areas.sum(): {voronoi_areas.sum()}")

# type of each constraint (0 for spring, 1 for bending)
n_constraints = 0
max_constraints = len(edges) + n_particles
constraint_types_np = np.zeros(max_constraints, dtype=np.int32)
# number of particles involved in each constraint
constraint_sizes_np = np.zeros(max_constraints, dtype=np.int32)
# indices of particles involved in each constraint
constraint_indices_np = np.zeros((max_constraints, 9), dtype=np.int32)
# AS in the PD paper
constraint_A_np = np.zeros((max_constraints, 9), dtype=np.float32)
# rest length of each spring; or rest mean curvature of each bending
constraint_params_np = np.zeros(max_constraints, dtype=np.float32)
constraint_weights_np = np.zeros(max_constraints, dtype=np.float32)

# For springs, AS[e] = [... 1, ..., -1, ...], at the position of the two particles
AS_spring_sp = sp.coo_matrix(
    (
        np.tile(
            np.array([1.0, -1.0], dtype=np.float32), len(edges)
        ),  # [1, -1, 1, -1, ...]
        (
            np.arange(len(edges)).repeat(2),  # [0, 0, 1, 1, ...]
            np.array(edges, dtype=np.int32).reshape(-1),  # [u0, v0, u1, v1, ...]
        ),
    ),
    shape=(len(edges), n_particles),
).tocsr()
for edge in edges:
    i = n_constraints
    constraint_types_np[i] = 0
    constraint_sizes_np[i] = 2
    constraint_indices_np[i, :2] = edge
    constraint_A_np[i, :2] = np.array([1, -1], dtype=np.float32)
    constraint_params_np[i] = np.linalg.norm(vertices[edge[0]] - vertices[edge[1]])
    constraint_weights_np[i] = spring_stiffness
    n_constraints += 1

# For bending, AS = -(cotmatrix / voronoi_areas[:, None])
AS_bend_sp = -(cotmatrix / voronoi_areas[:, None])
AS_bend_sp = AS_bend_sp.tocsr()
# AS_bend_sp[boundary_vertices] *= 0.0
rest_mean_curvatures = AS_bend_sp.dot(vertices)

cotmatrix_lil = cotmatrix.tolil()
for row in range(n_particles):
    i = n_constraints
    constraint_types_np[i] = 1
    constraint_size = len(cotmatrix_lil.rows[row])
    constraint_sizes_np[i] = constraint_size
    constraint_indices_np[i, :constraint_size] = cotmatrix_lil.rows[row]
    constraint_A_np[i, :constraint_size] = (
        -np.array(cotmatrix_lil.data[row], dtype=np.float32) / voronoi_areas[row]
    )
    constraint_params_np[i] = np.linalg.norm(rest_mean_curvatures[row])
    constraint_weights_np[i] = bending_stiffness * voronoi_areas[row]
    # if is_boundary[row]:
    #     constraint_weights_np[i] = 0.0
    n_constraints += 1

# set masks
masks_np = np.ones(n_particles, dtype=np.float32)
x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
for i in range(n_particles):
    x, y, z = vertices[i]
    if x == x_min and (y == y_min or y == y_max):
        masks_np[i] = 0.0


masses_np = voronoi_areas * cloth_thickness * cloth_density

######## Initialize PD variables ########
q_rest = wp.array(vertices, dtype=wp.vec3, device=device)
q = wp.array(
    vertices + np.array([0.0, 0.0, init_height], dtype=np.float32),
    dtype=wp.vec3,
    device=device,
)
q_prev_step = wp.zeros(n_particles, dtype=wp.vec3, device=device)
qd_prev_step = wp.zeros(n_particles, dtype=wp.vec3, device=device)
qd = wp.zeros(n_particles, dtype=wp.vec3, device=device)
masks = wp.array(masks_np, dtype=wp.float32, device=device)
masses = wp.array(masses_np, dtype=wp.float32, device=device)
s = wp.zeros(n_particles, dtype=wp.vec3, device=device)
b = wp.zeros(n_particles, dtype=wp.vec3, device=device)
constraint_types = wp.array(constraint_types_np, dtype=wp.int32, device=device)
constraint_sizes = wp.array(constraint_sizes_np, dtype=wp.int32, device=device)
constraint_indices = wp.array(constraint_indices_np, dtype=wp.int32, device=device)
constraint_A = wp.array(constraint_A_np, dtype=wp.float32, device=device)
constraint_params = wp.array(constraint_params_np, dtype=wp.float32, device=device)
constraint_weights = wp.array(constraint_weights_np, dtype=wp.float32, device=device)
P_diag = wp.zeros(n_particles, dtype=wp.float32, device=device)
qd_next_iter = wp.zeros(n_particles, dtype=wp.vec3, device=device)  # v^(k+1)
qd_prev_iter = wp.zeros(n_particles, dtype=wp.vec3, device=device)  # v^(k-1)
f_wo_contact = wp.zeros(
    n_particles, dtype=wp.vec3, device=device
)  # f^{k} in the PD fric paper
f_contact = wp.zeros(
    n_particles, dtype=wp.vec3, device=device
)  # xi(r) in the PD fric paper
q_last_ccd = wp.zeros(n_particles, dtype=wp.vec3, device=device)
ccd_step = wp.zeros(1, dtype=wp.float32, device=device)
# compute error history
error = wp.zeros(n_particles, dtype=wp.vec3, device=device)

######## Initialize kinematic targets ########
target_mask_np = np.zeros(n_particles, dtype=np.float32)
target_q_np = np.zeros((n_particles, 3), dtype=np.float32)
target_mask = wp.array(target_mask_np, dtype=wp.float32, device=device)
target_q = wp.array(target_q_np, dtype=wp.vec3, device=device)


######## Initialize boundary conditions (planes) ########
n_planes = 1
plane_normals_np = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
plane_offsets_np = np.array([0.0], dtype=np.float32)
plane_normals = wp.array(plane_normals_np, dtype=wp.vec3, device=device)
plane_offsets = wp.array(plane_offsets_np, dtype=wp.float32, device=device)


######## Add boundary spheres (a gripper consisting of 2 spheres ########
gripper_distance = 0.2
gripper_radius = 0.05
gripper_center = np.array([0.0, 0.0, 0.6], dtype=np.float32)
gripper_q = wp.array(
    [
        gripper_center + np.array([-gripper_distance / 2, 0.0, 0.0]),
        gripper_center + np.array([gripper_distance / 2, 0.0, 0.0]),
    ],
    dtype=wp.vec3,
    device=device,
)
gripper_qd = wp.zeros(2, dtype=wp.vec3, device=device)
if gripper_shape == "box":
    boundary_boxes_rest = wp.array(
        [
            [
                [-gripper_radius, -gripper_radius, -gripper_radius],
                [gripper_radius, gripper_radius, gripper_radius],
            ],
            [
                [-gripper_radius, -gripper_radius, -gripper_radius],
                [gripper_radius, gripper_radius, gripper_radius],
            ],
        ],
        dtype=wp.vec3,
        device=device,
    )


######## Initialize collision arrays ########
collision_P = wp.zeros(n_particles, dtype=wp.mat33, device=device)
collision_b = wp.zeros(n_particles, dtype=wp.vec3, device=device)

######## Initialize contact ########
hash_grid = wp.HashGrid(
    dim_x=hash_grid_dim, dim_y=hash_grid_dim, dim_z=hash_grid_dim, device=device
)


######## Simulation ########
@wp.kernel
def kinematic_update(
    qd: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    h: wp.float32,
    gravity: wp.vec3,
    velocity_damping: wp.float32,
    s: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    qd_i = qd[i] * (1.0 - velocity_damping) + gravity * h * masks[i]
    qd[i] = qd_i
    s[i] = qd_i


@wp.kernel
def init_b(
    s: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    masses: wp.array(dtype=wp.float32),
    h: wp.float32,
    b: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    # Linear system of v:
    #     P * v = b
    # Init b = M * s, where s = (q_target - q_prev) / h
    b_i = masses[i] * s[i] * masks[i]
    b[i] = b_i


@wp.kernel
def init_b_with_P_q_prev(
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    constraint_sizes: wp.array(dtype=wp.int32),
    constraint_indices: wp.array2d(dtype=wp.int32),
    constraint_A: wp.array2d(dtype=wp.float32),
    constraint_weights: wp.array(dtype=wp.float32),
    h: wp.float32,
    b: wp.array(dtype=wp.vec3),
):
    con_id, i_local, j_local = wp.tid()
    con_size = constraint_sizes[con_id]
    if i_local < con_size and j_local < con_size:
        i = constraint_indices[con_id, i_local]
        j = constraint_indices[con_id, j_local]
        if masks[i] != 0.0:
            coeff = (
                constraint_A[con_id, i_local]
                * constraint_A[con_id, j_local]
                * constraint_weights[con_id]
                * h
            )
            b_upd = coeff * q_prev[j]
            wp.atomic_sub(b, i, b_upd)


@wp.kernel
def project_Bp_elastic(
    q: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    constraint_sizes: wp.array(dtype=wp.int32),
    constraint_indices: wp.array2d(dtype=wp.int32),
    constraint_A: wp.array2d(dtype=wp.float32),
    constraint_params: wp.array(dtype=wp.float32),
    constraint_weights: wp.array(dtype=wp.float32),
    h: wp.float32,
    b: wp.array(dtype=wp.vec3),
):
    con_id = wp.tid()

    Aq = wp.vec3(0.0)
    for i_local in range(constraint_sizes[con_id]):
        i = constraint_indices[con_id, i_local]
        Aq += constraint_A[con_id, i_local] * q[i]

    Bp = wp.normalize(Aq) * constraint_params[con_id]

    for i_local in range(constraint_sizes[con_id]):
        i = constraint_indices[con_id, i_local]
        if masks[i] != 0.0:
            b_upd = Bp * (
                constraint_A[con_id, i_local] * constraint_weights[con_id] * h
            )
            wp.atomic_add(b, i, b_upd)


@wp.kernel
def jacobi_compute_f_wo_contact(
    qd: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    constraint_sizes: wp.array(dtype=wp.int32),
    constraint_indices: wp.array2d(dtype=wp.int32),
    constraint_A: wp.array2d(dtype=wp.float32),
    constraint_weights: wp.array(dtype=wp.float32),
    h: wp.float32,
    f_wo_contact: wp.array(dtype=wp.vec3),
):
    con_id, i_local, j_local = wp.tid()
    # j_local = wp.select(j_local >= i_local, j_local, j_local + 1)  # skip diagonal

    if i_local == j_local:
        return

    con_size = constraint_sizes[con_id]
    if i_local < con_size and j_local < con_size:
        i = constraint_indices[con_id, i_local]
        j = constraint_indices[con_id, j_local]
        if masks[i] != 0.0:
            wp.atomic_sub(
                f_wo_contact,
                i,
                (
                    constraint_A[con_id, i_local]
                    * constraint_A[con_id, j_local]
                    * constraint_weights[con_id]
                    * (h * h)
                )
                * qd[j],
            )


@wp.func
def mat33_diag_vec3(a: wp.mat33):
    return wp.vec3(a[0, 0], a[1, 1], a[2, 2])


@wp.func
def mat33_off_diag(a: wp.mat33):
    return wp.mat33(0.0, a[0, 1], a[0, 2], a[1, 0], 0.0, a[1, 2], a[2, 0], a[2, 1], 0.0)


@wp.kernel
def jacobi_compute_f_contact_pg(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=wp.float32),
    collision_weight: wp.float32,
    collision_sphere_radius: wp.float32,
    h: wp.float32,
    d_hat: wp.float32,
    friction_mu: wp.float32,
    sor_factor: wp.float32,
    f_wo_contact: wp.array(dtype=wp.vec3),
    f_contact: wp.array(dtype=wp.vec3),
    collision_P: wp.array(dtype=wp.mat33),
):
    particle_i, plane_id = wp.tid()
    x = q[particle_i]
    v = qd[particle_i]
    x_prev = q_prev[particle_i]
    n = plane_normals[plane_id]
    of = plane_offsets[plane_id]
    d = wp.dot(x, n) - of - collision_sphere_radius
    f_wo_contact_i = f_wo_contact[particle_i]
    if d < d_hat:
        col_A = wp.outer(n, n)
        col_Bp = x + n * (d_hat - d)
        col_b = col_A * (col_Bp - x_prev) * collision_weight * h
        col_P = col_A * collision_weight * h * h

        f_contact_offdiag_i = col_b - mat33_off_diag(col_P) * v
        f_contact_i = col_b - col_P * v
        f_fric_i = wp.vec3(0.0)

        f_wo_contact_N = wp.dot(f_wo_contact_i, n)
        f_wo_contact_T = f_wo_contact_i - f_wo_contact_N * n
        if f_wo_contact_N < 0:
            f_contact_i_norm = wp.max(wp.dot(f_contact_i, n), 0.0)
            f_fric_i = (
                -wp.normalize(f_wo_contact_T)
                * wp.min(wp.length(f_wo_contact_T), friction_mu * f_contact_i_norm)
                * sor_factor
            )

        wp.atomic_add(f_contact, particle_i, f_contact_offdiag_i + f_fric_i)
        wp.atomic_add(collision_P, particle_i, col_P)


@wp.kernel
def jacobi_compute_f_contact_pp_hashgrid(
    grid: wp.uint64,
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    q_rest: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=wp.float32),
    collision_sphere_radius: wp.float32,
    h: wp.float32,
    d_hat: wp.float32,
    friction_mu: wp.float32,
    sor_factor: wp.float32,
    f_wo_contact: wp.array(dtype=wp.vec3),
    P_diag: wp.array(dtype=wp.float32),
    f_contact: wp.array(dtype=wp.vec3),
    collision_P: wp.array(dtype=wp.mat33),
):
    i0 = wp.tid()
    x0 = q[i0]
    v0 = qd[i0]
    x0_prev = q_prev[i0]
    m0 = mass[i0]

    query = wp.hash_grid_query(grid, x0, d_hat + 2.0 * collision_sphere_radius)
    i1 = int(0)

    while wp.hash_grid_query_next(query, i1):
        if i0 < i1:
            x1 = q[i1]
            v1 = qd[i1]
            x1_prev = q_prev[i1]
            m1 = mass[i1]
            n = wp.normalize(x0 - x1)
            d = wp.length(x0 - x1) - 2.0 * collision_sphere_radius
            d_rest = wp.length(q_rest[i0] - q_rest[i1]) - 2.0 * collision_sphere_radius
            if d < d_hat and d_rest >= d_hat:
                I3 = wp.identity(n=3, dtype=float)
                delta_x = n * (d_hat - d)

                mass_scale0 = m1 / (m0 + m1)
                mass_scale1 = m0 / (m0 + m1)
                # collision_weight0 = collision_weight
                # collision_weight1 = collision_weight
                collision_weight0 = collision_weight / mass_scale0
                collision_weight1 = collision_weight / mass_scale1

                delta_x0 = delta_x * mass_scale0
                delta_x1 = -delta_x * mass_scale1
                col_Bp0 = x0 + delta_x0
                col_Bp1 = x1 + delta_x1
                # col_Bp0 = x0 + delta_x / 2.0
                # col_Bp1 = x1 - delta_x / 2.0

                col_A = wp.outer(n, n)
                col_b0 = col_A * (col_Bp0 - x0_prev) * collision_weight0 * h
                col_b1 = col_A * (col_Bp1 - x1_prev) * collision_weight1 * h
                col_P0 = col_A * collision_weight0 * h * h
                col_P1 = col_A * collision_weight1 * h * h

                f_wo_contact0 = f_wo_contact[i0]
                f_wo_contact1 = f_wo_contact[i1]
                f_contact_offdiag0 = col_b0 - mat33_off_diag(col_P0) * v0
                f_contact_offdiag1 = col_b1 - mat33_off_diag(col_P1) * v1
                f_contact0 = h * collision_weight * delta_x
                f_contact1 = -f_contact0
                # f_contact0 = col_b0 - col_P0 * v0
                # f_contact1 = col_b1 - col_P1 * v1  # f_contact1 = -f_contact0

                f_fric0 = wp.vec3(0.0)
                f_fric1 = wp.vec3(0.0)

                v0_T = v0 - wp.dot(v0, n) * n
                v1_T = v1 - wp.dot(v1, n) * n
                f_wo_contact_N0 = wp.dot(f_wo_contact0, n)
                f_wo_contact_N1 = wp.dot(f_wo_contact1, -n)
                f_wo_contact_T0 = f_wo_contact0 - f_wo_contact_N0 * n
                f_wo_contact_T1 = f_wo_contact1 + f_wo_contact_N1 * n
                D0 = wp.vec3(P_diag[i0]) + mat33_diag_vec3(col_P0)
                D1 = wp.vec3(P_diag[i1]) + mat33_diag_vec3(col_P1)
                f_wo_contact_T0 -= wp.cw_mul(D0, v1_T)
                f_wo_contact_T1 -= wp.cw_mul(D1, v0_T)
                f_contact0_N = wp.max(wp.dot(f_contact0, n), 0.0)
                f_fric0 = (
                    -wp.normalize(f_wo_contact_T0)
                    * wp.min(wp.length(f_wo_contact_T0), friction_mu * f_contact0_N)
                    * sor_factor
                )
                f_contact1_N = wp.max(wp.dot(f_contact1, -n), 0.0)
                f_fric1 = (
                    -wp.normalize(f_wo_contact_T1)
                    * wp.min(wp.length(f_wo_contact_T1), friction_mu * f_contact1_N)
                    * sor_factor
                )

                wp.atomic_add(f_contact, i0, f_contact_offdiag0 + f_fric0)
                wp.atomic_add(f_contact, i1, f_contact_offdiag1 + f_fric1)
                wp.atomic_add(collision_P, i0, col_P0)
                wp.atomic_add(collision_P, i1, col_P1)

                # f0 = f_wo_contact[i0]
                # f1 = f_wo_contact[i1]
                # f0_N = wp.dot(f0, n)
                # f1_N = wp.dot(f1, n)
                # D0 = P_diag[i0]
                # D1 = P_diag[i1]
                # r0_N = (D0 * f1_N - D1 * f0_N) / (D0 + D1)
                # if r0_N > 0:
                #     r0 = r0_N * n
                #     wp.atomic_add(f_contact, i0, r0)
                #     wp.atomic_add(f_contact, i1, -r0)


@wp.kernel
def jacobi_compute_f_contact_pp_boundary(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    gripper_q: wp.array(dtype=wp.vec3),
    gripper_qd: wp.array(dtype=wp.vec3),
    gripper_radius: wp.float32,
    collision_weight: wp.float32,
    collision_sphere_radius: wp.float32,
    h: wp.float32,
    d_hat: wp.float32,
    friction_mu: wp.float32,
    sor_factor: wp.float32,
    f_wo_contact: wp.array(dtype=wp.vec3),
    P_diag: wp.array(dtype=wp.float32),
    f_contact: wp.array(dtype=wp.vec3),
    collision_P: wp.array(dtype=wp.mat33),
):
    i0, i1 = wp.tid()
    x0 = q[i0]
    v0 = qd[i0]
    x0_prev = q_prev[i0]
    x1 = gripper_q[i1]
    v1 = gripper_qd[i1]
    n = wp.normalize(x0 - x1)
    d = wp.length(x0 - x1) - gripper_radius - collision_sphere_radius
    f_wo_contact_i = f_wo_contact[i0]
    if d < d_hat:
        col_A = wp.outer(n, n)
        col_Bp = x0 + n * (d_hat - d)
        col_b = col_A * (col_Bp - x0_prev) * collision_weight * h
        col_P = col_A * collision_weight * h * h

        f_contact_offdiag_i = col_b - mat33_off_diag(col_P) * v0
        f_contact_i = col_b - col_P * v0
        f_fric_i = wp.vec3(0.0)

        v1_T = v1 - wp.dot(v1, n) * n

        f_wo_contact_N = wp.dot(f_wo_contact_i, n)
        f_wo_contact_T = f_wo_contact_i - f_wo_contact_N * n
        f_wo_contact_T -= wp.cw_mul(
            wp.vec3(P_diag[i0]) + mat33_diag_vec3(collision_P[i0]), v1_T
        )
        if f_wo_contact_N < 0:
            f_contact_i_norm = wp.max(wp.dot(f_contact_i, n), 0.0)
            f_fric_i = (
                -wp.normalize(f_wo_contact_T)
                * wp.min(wp.length(f_wo_contact_T), friction_mu * f_contact_i_norm)
                * sor_factor
            )

        wp.atomic_add(f_contact, i0, f_contact_offdiag_i + f_fric_i)
        wp.atomic_add(collision_P, i0, col_P)


@wp.func
def check_inside_box(
    p: wp.vec3,
    x_min: wp.vec3,
    x_max: wp.vec3,
):
    return (
        p[0] >= x_min[0]
        and p[0] <= x_max[0]
        and p[1] >= x_min[1]
        and p[1] <= x_max[1]
        and p[2] >= x_min[2]
        and p[2] <= x_max[2]
    )


@wp.func
def project_interior_onto_box(
    p: wp.vec3,
    x_min: wp.vec3,
    x_max: wp.vec3,
):
    # project p that is inside the box to the boundary
    p_proj = wp.vec3(0.0)
    min_dist = 1e9
    for i in range(3):
        # check min
        dist = p[i] - x_min[i]
        if dist < min_dist:
            min_dist = dist
            p_proj = p
            p_proj[i] = x_min[i]
        # check max
        dist = x_max[i] - p[i]
        if dist < min_dist:
            min_dist = dist
            p_proj = p
            p_proj[i] = x_max[i]
    return p_proj


@wp.func
def project_exterior_onto_box(
    p: wp.vec3,
    x_min: wp.vec3,
    x_max: wp.vec3,
):
    # project p that is outside the box to the boundary
    return wp.vec3(
        wp.clamp(p[0], x_min[0], x_max[0]),
        wp.clamp(p[1], x_min[1], x_max[1]),
        wp.clamp(p[2], x_min[2], x_max[2]),
    )


@wp.kernel
def jacobi_compute_f_contact_pbox_boundary(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    boundary_boxes_rest: wp.array(dtype=wp.vec3, ndim=2),
    gripper_q: wp.array(dtype=wp.vec3),
    gripper_qd: wp.array(dtype=wp.vec3),
    collision_weight: wp.float32,
    collision_sphere_radius: wp.float32,
    h: wp.float32,
    d_hat: wp.float32,
    friction_mu: wp.float32,
    sor_factor: wp.float32,
    f_wo_contact: wp.array(dtype=wp.vec3),
    P_diag: wp.array(dtype=wp.float32),
    f_contact: wp.array(dtype=wp.vec3),
    collision_P: wp.array(dtype=wp.mat33),
):
    i0, i1 = wp.tid()
    x0 = q[i0]
    v0 = qd[i0]
    x0_prev = q_prev[i0]

    x_min = gripper_q[i1] + boundary_boxes_rest[i1, 0]
    x_max = gripper_q[i1] + boundary_boxes_rest[i1, 1]
    v1 = gripper_qd[i1]
    if check_inside_box(x0, x_min, x_max):
        x1 = project_interior_onto_box(x0, x_min, x_max)
        n = -wp.normalize(x0 - x1)
        d = -wp.length(x0 - x1) - collision_sphere_radius
    else:
        x1 = project_exterior_onto_box(x0, x_min, x_max)
        n = wp.normalize(x0 - x1)
        d = wp.length(x0 - x1) - collision_sphere_radius

    f_wo_contact_i = f_wo_contact[i0]
    if d < d_hat:
        col_A = wp.outer(n, n)
        col_Bp = x0 + n * (d_hat - d)
        col_b = col_A * (col_Bp - x0_prev) * collision_weight * h
        col_P = col_A * collision_weight * h * h

        f_contact_offdiag_i = col_b - mat33_off_diag(col_P) * v0
        f_contact_i = col_b - col_P * v0
        f_fric_i = wp.vec3(0.0)

        v1_T = v1 - wp.dot(v1, n) * n

        f_wo_contact_N = wp.dot(f_wo_contact_i, n)
        f_wo_contact_T = f_wo_contact_i - f_wo_contact_N * n
        f_wo_contact_T -= wp.cw_mul(
            wp.vec3(P_diag[i0]) + mat33_diag_vec3(collision_P[i0]), v1_T
        )

        if f_wo_contact_N < 0:
            f_contact_i_norm = wp.max(wp.dot(f_contact_i, n), 0.0)
            f_fric_i = (
                -wp.normalize(f_wo_contact_T)
                * wp.min(wp.length(f_wo_contact_T), friction_mu * f_contact_i_norm)
                * sor_factor
            )

        wp.atomic_add(f_contact, i0, f_contact_offdiag_i + f_fric_i)
        wp.atomic_add(collision_P, i0, col_P)


@wp.kernel
def jacobi_step_with_f_contact(
    f_wo_contact: wp.array(dtype=wp.vec3),
    f_contact: wp.array(dtype=wp.vec3),
    P_diag: wp.array(dtype=wp.float32),
    collision_P: wp.array(dtype=wp.mat33),
    masks: wp.array(dtype=wp.float32),
    qd_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if masks[i] != 0.0:
        P_diag_i = wp.vec3(P_diag[i]) + mat33_diag_vec3(collision_P[i])
        qd_out[i] = wp.cw_div(f_contact[i] + f_wo_contact[i], P_diag_i)


@wp.kernel
def ccd_pg(
    q: wp.array(dtype=wp.vec3),
    q_last_ccd: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=wp.float32),
    collision_sphere_radius: wp.float32,
    ccd_slackness: wp.float32,
    ccd_thickness: wp.float32,
    ccd_step: wp.array(dtype=wp.float32),
):
    particle_i, plane_id = wp.tid()
    x = q_last_ccd[particle_i]
    v = q[particle_i] - x
    n = plane_normals[plane_id]
    of = plane_offsets[plane_id]
    d = wp.dot(x, n) - of - collision_sphere_radius
    if d > 0:
        if d < ccd_thickness:
            wp.atomic_min(ccd_step, 0, 0.0)
            wp.printf("[ccd_pg] d = %.3e < %.3e\n", d, ccd_thickness)
        t_impact = (d - ccd_thickness) / -wp.dot(v, n) * ccd_slackness
        # wp.printf("d = %.3e > 0, t_impact = %.3e\n", d, t_impact)
        if t_impact >= 0.0 and t_impact < 1.0:
            wp.atomic_min(ccd_step, 0, t_impact)


@wp.kernel
def ccd_pp_hashgrid(
    grid: wp.uint64,
    q: wp.array(dtype=wp.vec3),
    q_rest: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    collision_sphere_radius: wp.float32,
    ccd_slackness: wp.float32,
    ccd_thickness: wp.float32,
    d_hat: wp.float32,
    ccd_step: wp.array(dtype=wp.float32),
):
    i0 = wp.tid()
    x0 = q[i0]

    query = wp.hash_grid_query(
        grid, x0, d_hat + 2.0 * collision_sphere_radius + ccd_thickness
    )
    i1 = int(0)

    min_t_impact = float(1.0)

    while wp.hash_grid_query_next(query, i1):
        x01 = q_prev[i1] - q_prev[i0]
        v01 = q[i1] - q[i0] - x01
        d_t0 = wp.length(x01) - 2.0 * collision_sphere_radius
        d_rest = wp.length(q_rest[i1] - q_rest[i0]) - 2.0 * collision_sphere_radius
        if d_rest <= d_hat:
            return
        if d_t0 > 0:
            if d_t0 <= ccd_thickness:
                min_t_impact = 0.0
            else:
                # (x + t * v)^2 <= (2 * r - thickness)^2
                # <=> dot(v, v) * t^2 + 2 * dot(x, v) * t + dot(x, x) - (2 * r - thickness)^2 <= 0
                a = wp.dot(v01, v01)
                b = 2.0 * wp.dot(x01, v01)
                c = (
                    wp.dot(x01, x01)
                    - (2.0 * collision_sphere_radius - ccd_thickness) ** 2.0
                )
                t_impact = float(1.0)
                if wp.abs(a) < 1e-6:
                    t_impact = -c / b * ccd_slackness
                else:
                    discriminant = b * b - 4.0 * a * c
                    if discriminant >= 0.0:
                        t_impact = (
                            (-b - wp.sqrt(discriminant)) / (2.0 * a) * ccd_slackness
                        )
                if t_impact >= 0.0 and t_impact < 1.0:
                    min_t_impact = wp.min(min_t_impact, t_impact)

            # if min_t_impact == 0.0:
            #     wp.printf("min_t_impact = 0.0, d_t0 = %.3e\n", d_t0)

    wp.atomic_min(ccd_step, 0, min_t_impact)


@wp.kernel
def step_q_after_ccd(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    h: wp.float32,
    q_last_ccd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    ccd_step: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    q[i] = q_last_ccd[i] + (q[i] - q_last_ccd[i]) * ccd_step[0]
    qd[i] = (q[i] - q_prev[i]) / h
    q_last_ccd[i] = q[i]


@wp.kernel
def init_P_diag(
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.float32),
    h: wp.float32,
    P_diag: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    P_diag[i] = masses[i] * masks[i]


@wp.kernel
def compute_P_diag(
    masks: wp.array(dtype=wp.float32),
    constraint_sizes: wp.array(dtype=wp.int32),
    constraint_indices: wp.array2d(dtype=wp.int32),
    constraint_A: wp.array2d(dtype=wp.float32),
    constraint_weights: wp.array(dtype=wp.float32),
    h: wp.float32,
    P_diag: wp.array(dtype=wp.float32),
):
    # Without collision A
    con_id, i_local = wp.tid()
    con_size = constraint_sizes[con_id]
    if i_local < con_size:
        i = constraint_indices[con_id, i_local]
        if masks[i] != 0.0:
            diag_upd = constraint_A[con_id, i_local]
            wp.atomic_add(
                P_diag, i, diag_upd * diag_upd * constraint_weights[con_id] * h * h
            )


@wp.kernel
def compute_q_from_qd(
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    h: wp.float32,
    q: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    q[i] = q_prev[i] + qd[i] * h


@wp.kernel
def update_qd_chebyshev(
    qd: wp.array(dtype=wp.vec3),
    qd_next: wp.array(dtype=wp.vec3),  # q^(k+1)
    qd_last: wp.array(dtype=wp.vec3),  # q^(k-1)
    omega: wp.float32,
    gamma: wp.float32,
):
    i = wp.tid()
    q_next_i = qd_next[i]
    q_last_i = qd_last[i]
    q_i = qd[i]
    qd[i] = omega * (gamma * (q_next_i - q_i) + q_i - q_last_i) + q_last_i
    qd_last[i] = q_i


wp.launch(
    kernel=init_P_diag,
    dim=n_particles,
    inputs=[masses, masks, time_step],
    outputs=[P_diag],
    device=device,
)
wp.launch(
    kernel=compute_P_diag,
    dim=(n_constraints, 9),
    inputs=[
        masks,
        constraint_sizes,
        constraint_indices,
        constraint_A,
        constraint_weights,
        time_step,
        P_diag,
    ],
    device=device,
)


def sim_step():
    if debug_nan_flag:
        assert not np.any(np.isnan(q.numpy()))

    if use_hash_grid:
        hash_grid.build(
            q,
            hash_grid_size_scale,
        )

    wp.copy(q_prev_step, q)
    wp.copy(qd_prev_step, qd)
    wp.copy(q_last_ccd, q)

    omega = 1.0

    # Compute s; init q = s
    wp.launch(
        kernel=kinematic_update,
        dim=n_particles,
        inputs=[qd, masks, time_step, gravity, velocity_damping, s],
        device=device,
    )
    wp.copy(qd_prev_iter, qd)

    if n_pd_iters % ccd_every != 0:
        print(
            f"Warning: n_pd_iters is not divisible by ccd_every. Assuming n_pd_iters = (n_pd_iters // ccd_every) * ccd_every = {n_pd_iters // ccd_every * ccd_every}."
        )

    for pd_step in range(n_pd_iters):
        wp.launch(
            kernel=init_b,
            dim=n_particles,
            inputs=[s, masks, masses, time_step, b],
            device=device,
        )
        wp.launch(
            kernel=init_b_with_P_q_prev,
            dim=(n_constraints, 9, 9),
            inputs=[
                qd_prev_iter,
                q_prev_step,
                masks,
                constraint_sizes,
                constraint_indices,
                constraint_A,
                constraint_weights,
                time_step,
                b,
            ],
            device=device,
        )
        wp.launch(
            kernel=project_Bp_elastic,
            dim=n_constraints,
            inputs=[
                q,
                q_prev_step,
                masks,
                constraint_sizes,
                constraint_indices,
                constraint_A,
                constraint_params,
                constraint_weights,
                time_step,
                b,
            ],
            device=device,
        )
        wp.copy(f_wo_contact, b)

        wp.launch(
            kernel=jacobi_compute_f_wo_contact,
            dim=(n_constraints, 9, 9),
            inputs=[
                qd,
                masks,
                constraint_sizes,
                constraint_indices,
                constraint_A,
                constraint_weights,
                time_step,
                f_wo_contact,
            ],
            device=device,
        )

        # Compute contact forces
        f_contact.zero_()
        collision_P.zero_()
        wp.launch(
            kernel=jacobi_compute_f_contact_pg,
            dim=(n_particles, n_planes),
            inputs=[
                q,
                qd,
                q_prev_step,
                plane_normals,
                plane_offsets,
                collision_weight,
                collision_sphere_radius,
                time_step,
                d_hat,
                ground_friction_mu,
                sor_factor,
                f_wo_contact,
                f_contact,
                collision_P,
            ],
            device=device,
        )
        if gripper_shape == "sphere":
            wp.launch(
                kernel=jacobi_compute_f_contact_pp_boundary,
                dim=(n_particles, 2),
                inputs=[
                    q,
                    qd,
                    q_prev_step,
                    gripper_q,
                    gripper_qd,
                    gripper_radius,
                    collision_weight,
                    collision_sphere_radius,
                    time_step,
                    d_hat,
                    gripper_friction_mu,
                    sor_factor,
                    f_wo_contact,
                    P_diag,
                    f_contact,
                    collision_P,
                ],
                device=device,
            )
        elif gripper_shape == "box":
            wp.launch(
                kernel=jacobi_compute_f_contact_pbox_boundary,
                dim=(n_particles, 2),
                inputs=[
                    q,
                    qd,
                    q_prev_step,
                    boundary_boxes_rest,
                    gripper_q,
                    gripper_qd,
                    collision_weight,
                    collision_sphere_radius,
                    time_step,
                    d_hat,
                    gripper_friction_mu,
                    sor_factor,
                    f_wo_contact,
                    P_diag,
                    f_contact,
                    collision_P,
                ],
                device=device,
            )
        if use_hash_grid:
            wp.launch(
                kernel=jacobi_compute_f_contact_pp_hashgrid,
                dim=n_particles,
                inputs=[
                    hash_grid.id,
                    q,
                    qd,
                    q_prev_step,
                    q_rest,
                    masses,
                    collision_sphere_radius,
                    time_step,
                    d_hat,
                    cloth_friction_mu,
                    sor_factor,
                    f_wo_contact,
                    P_diag,
                    f_contact,
                    collision_P,
                ],
                device=device,
            )
        else:
            raise NotImplementedError

        wp.launch(
            kernel=jacobi_step_with_f_contact,
            dim=n_particles,
            inputs=[f_wo_contact, f_contact, P_diag, collision_P, masks, qd_next_iter],
            device=device,
        )

        if use_chebyshev:
            if pd_step < chebyshev_n_warmup_steps:
                omega = 1.0
            elif pd_step == chebyshev_n_warmup_steps:
                omega = 2.0 / (2.0 - pd_rho**2)
            else:
                omega = 4.0 / (4.0 - pd_rho**2 * omega)
            wp.launch(
                kernel=update_qd_chebyshev,
                dim=n_particles,
                inputs=[qd, qd_next_iter, qd_prev_iter, omega, pd_gamma],
                device=device,
            )
        else:
            wp.copy(qd_prev_iter, qd)
            wp.copy(qd, qd_next_iter)

        wp.launch(
            kernel=compute_q_from_qd,
            dim=n_particles,
            inputs=[qd, q_prev_step, time_step, q],
            device=device,
        )

        if (pd_step + 1) % ccd_every == 0 or pd_step == n_pd_iters - 1:
            ccd_step.fill_(1.0)
            if ccd_flag:
                wp.launch(
                    kernel=ccd_pg,
                    dim=(n_particles, n_planes),
                    inputs=[
                        q,
                        q_last_ccd,
                        masks,
                        plane_normals,
                        plane_offsets,
                        collision_sphere_radius,
                        ccd_slackness,
                        ccd_thickness,
                    ],
                    outputs=[ccd_step],
                    device=device,
                )
            # print(f"ccd_step = {ccd_step.numpy()[0]}")

            wp.launch(
                kernel=step_q_after_ccd,
                dim=n_particles,
                inputs=[q, qd, time_step, q_last_ccd, q_prev_step, ccd_step],
                device=device,
            )

    wp.launch(
        kernel=compute_q_from_qd,
        dim=2,
        inputs=[gripper_qd, gripper_q, time_step, gripper_q],
        device=device,
    )
    gripper_qd.zero_()


######## Simple renderer ########
scene = sapien.Scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([1, 0, -1], [1, 1, 1], True)
# scene.set_environment_map(os.path.join(assets_dir, "env.ktx"))
scene.add_ground(0.0)

# add a camera to indicate shader
cam_entity = sapien.Entity()
cam = sapien.render.RenderCameraComponent(512, 512)
cam.set_near(1e-3)
cam.set_far(100)
cam_entity.add_component(cam)
cam_entity.name = "camera"
cam_entity.set_pose(
    # sapien.Pose(
    #     [-1.0058, 0.161549, 0.75055], [0.957709, -0.0415881, 0.219196, 0.181705]
    # )
    # sapien.Pose(
    #     [-0.000511355, -0.581108, 0.740754], [0.73446, -0.146693, 0.1681, 0.64093]
    # )
    sapien.Pose([-0.064028, -0.270365, 0.440705], [0.73446, -0.146693, 0.1681, 0.64093])
)
scene.add_entity(cam_entity)

# entity and render component for the cloth
cloth_entity = sapien.Entity()
cloth_render = sapien.render.RenderCudaMeshComponent(len(vertices), 2 * len(faces))
cloth_render.set_vertex_count(len(vertices))
cloth_render.set_triangle_count(2 * len(faces))
cloth_render.set_triangles(np.concatenate([faces, faces[:, ::-1]], axis=0))
cloth_render.set_material(sapien.render.RenderMaterial(base_color=[0.7, 0.3, 0.4, 1.0]))
cloth_entity.add_component(cloth_render)
scene.add_entity(cloth_entity)

if render_vertices_flag:
    cloth_spheres = [sapien.Entity() for _ in range(len(vertices))]
    q_np = q.numpy()
    for i in range(len(vertices)):
        cloth_spheres[i].set_pose(sapien.Pose(q_np[i]))
        render_shape_sphere = sapien.render.RenderShapeSphere(
            collision_sphere_radius,
            sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.7, 0.8]),
        )
        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(render_shape_sphere)
        cloth_spheres[i].add_component(render_component)
    for i in range(len(vertices)):
        scene.add_entity(cloth_spheres[i])

if gripper_shape == "sphere":
    gripper_entities = [sapien.Entity() for _ in range(2)]
    boundary_spheres_np = gripper_q.numpy()
    for i in range(2):
        gripper_entities[i].set_pose(sapien.Pose(boundary_spheres_np[i]))
        render_shape_sphere = sapien.render.RenderShapeSphere(
            gripper_radius,
            sapien.render.RenderMaterial(base_color=[0.3, 0.7, 0.4, 0.8]),
        )
        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(render_shape_sphere)
        gripper_entities[i].add_component(render_component)
    for i in range(2):
        scene.add_entity(gripper_entities[i])
elif gripper_shape == "box":
    gripper_entities = [sapien.Entity() for _ in range(2)]
    boundary_boxes_np = gripper_q.numpy()
    boundary_boxes_rest_np = boundary_boxes_rest.numpy()
    for i in range(2):
        gripper_entities[i].set_pose(sapien.Pose(boundary_boxes_np[i]))
        render_shape_box = sapien.render.RenderShapeBox(
            (boundary_boxes_rest_np[i][1] - boundary_boxes_rest_np[i][0]) / 2.0,
            sapien.render.RenderMaterial(base_color=[0.3, 0.7, 0.4, 0.8]),
        )
        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(render_shape_box)
        gripper_entities[i].add_component(render_component)
    for i in range(2):
        scene.add_entity(gripper_entities[i])


def change_sphere_color(id, color):
    render_shape_sphere = sapien.render.RenderShapeSphere(
        collision_sphere_radius, sapien.render.RenderMaterial(base_color=color)
    )
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(render_shape_sphere)
    cloth_spheres[id].remove_component(cloth_spheres[id].get_components()[0])
    cloth_spheres[id].add_component(render_component)


@wp.kernel
def copy_positions_to_render(
    dst_vertices: wp.array2d(dtype=wp.float32),
    src_positions: wp.array(dtype=wp.vec3),
):
    i, j = wp.tid()
    dst_vertices[i, j] = src_positions[i][j]


def update_render_component(render_component: sapien.render.RenderCudaMeshComponent, q):
    interface = render_component.cuda_vertices.__cuda_array_interface__
    dst = wp.array(
        ptr=interface["data"][0],
        dtype=wp.float32,
        shape=interface["shape"],
        strides=interface["strides"],
        owner=False,
        device=q.device,
    )
    wp.launch(
        kernel=copy_positions_to_render,
        dim=(len(vertices), 3),
        inputs=[dst, q],
        device=q.device,
    )
    render_component.notify_vertex_updated(wp.get_stream(device).cuda_stream)

    if render_vertices_flag:
        q_np = q.numpy()
        for i in range(len(vertices)):
            cloth_spheres[i].set_pose(sapien.Pose(q_np[i]))
        # scene.update_render()

    gripper_q_np = gripper_q.numpy()
    for i in range(2):
        gripper_entities[i].set_pose(sapien.Pose(gripper_q_np[i]))
    scene.update_render()


gripper_keys = ["up", "down", "left", "right", "k", "i", "j", "l"]


def move_boundary_spheres(key, move_speed, grip_speed):
    global gripper_center, gripper_distance

    if key == "up":
        gripper_center[1] += move_speed
    elif key == "down":
        gripper_center[1] -= move_speed
    elif key == "left":
        gripper_center[0] -= move_speed
    elif key == "right":
        gripper_center[0] += move_speed
    elif key == "k":
        gripper_center[2] -= move_speed
        gripper_center[2] = max(
            gripper_radius + collision_sphere_radius, gripper_center[2]
        )
    elif key == "i":
        gripper_center[2] += move_speed
    elif key == "j":
        gripper_distance -= grip_speed
        gripper_distance = max(
            2 * gripper_radius + collision_sphere_radius, gripper_distance
        )
    elif key == "l":
        gripper_distance += grip_speed

    new_gripper_q = np.array(
        [
            gripper_center + np.array([-gripper_distance / 2, 0.0, 0.0]),
            gripper_center + np.array([gripper_distance / 2, 0.0, 0.0]),
        ],
        dtype=np.float32,
    )
    gripper_qd.assign((new_gripper_q - gripper_q.numpy()) / time_step)
    wp.synchronize()


viewer = sapien.utils.Viewer()
viewer.set_scene(scene)
viewer.set_camera_pose(cam.entity_pose)
viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)

output_dir = os.path.join(os.path.dirname(__file__), "output/cloth")
frames_dir = os.path.join(
    output_dir, f"frames_{'chebyshev' if use_chebyshev else 'vanilla'}"
)
states_dir = os.path.join(
    output_dir, f"states_{'chebyshev' if use_chebyshev else 'vanilla'}"
)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(states_dir, exist_ok=True)

if load_state:
    q_np, qd_np = np.load(state_path)
    q.assign(q_np)
    qd.assign(qd_np)

step = 0
viewer.paused = True
scene.update_render()
update_render_component(cloth_render, q)
viewer.render()

# change_sphere_color(0, [1.0, 0.8, 0.4, 0.8])

sim_step_cuda_graph = None
if use_cuda_graph:
    assert not compute_error_flag
    wp.capture_begin(device)
    for _ in range(render_every):
        sim_step()
    sim_step_cuda_graph = wp.capture_end(device)


start_time = time.time()

while not viewer.closed:
    for key in gripper_keys:
        if viewer.window.key_down(key):
            # print(f"Key {key} is pressed")
            move_boundary_spheres(
                key,
                move_speed=0.01,
                grip_speed=0.01,
            )

    if use_cuda_graph:
        wp.capture_launch(sim_step_cuda_graph)
    else:
        for _ in range(render_every):
            sim_step()

    if render_flag:
        scene.update_render()
        update_render_component(cloth_render, q)
        viewer.render()

    if save_render:
        assert render_flag
        cam.take_picture()
        rgba = cam.get_picture("Color")
        rgba = np.clip(rgba, 0, 1)[:, :, :3]
        rgba = Image.fromarray((rgba * 255).astype(np.uint8))
        rgba.save(os.path.join(frames_dir, f"step_{step:04d}.png"))

    if save_state:
        if step % 10 == 0:
            q_np = q.numpy()
            qd_np = qd.numpy()
            filename = f"step_{step:04d}.npy"
            np.save(os.path.join(states_dir, filename), (q_np, qd_np))

    step += 1

    if n_render_steps is not None and step >= n_render_steps:
        break

wp.synchronize()

end_time = time.time()
total_time = end_time - start_time

print(f"Total time: {total_time:.2f}s")
print(
    f"Wall time per step: {total_time / (step * render_every):.2e}s (simulated time: {time_step:.2e}s)"
)
print(
    f"Simulation speed = {time_step / (total_time / (step * render_every)):.2f} * real world speed"
)
