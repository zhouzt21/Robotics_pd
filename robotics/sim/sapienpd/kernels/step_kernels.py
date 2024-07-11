import warp as wp
from ..pd_defs import ShapeTypes


"""
Linear system of v:
    P v = b,
where
    P = M + h^2 Σ w_c A_c^T A_c,
    b = M s + h Σ w_c A_c^T (p_c - A_c q_prev).
Using Jacobi solver:
    f := b - R(P) v,
    v' = v + D(P)^{-1} f,
where D(P) is the diagonal of P and R(P) is the off-diagonal part of P.
"""


@wp.kernel
def kinematic_update(
    q: wp.array(dtype=wp.vec3),
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
    q[i] = q[i] + h * qd_i


@wp.kernel
def kinematic_update_body(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    h: wp.float32,
):
    i = wp.tid()
    X_wb_i = body_q[i]
    p = wp.transform_get_translation(X_wb_i)
    q = wp.transform_get_rotation(X_wb_i)
    V_i = body_qd[i]
    w = wp.spatial_top(V_i)
    v = wp.spatial_bottom(V_i)
    p_new = p + h * v
    q_new = q + h * wp.quat(w, 0.0) * q * 0.5
    q_new = wp.normalize(q_new)
    body_q[i] = wp.transform(p_new, q_new)


@wp.kernel
def init_f(
    s: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    masses: wp.array(dtype=wp.float32),
    f: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    # Init f = b = M s, where s = qd_target
    f_i = masses[i] * s[i] * masks[i]
    f[i] = f_i


@wp.kernel
def init_P_diag(
    masses: wp.array(dtype=wp.float32),
    damping: wp.float32,
    h: wp.float32,
    P_diag: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    P_diag[i] = masses[i] + damping


@wp.kernel
def compute_P_diag_elastic(
    elas_weights: wp.array(dtype=wp.float32),
    elas_params: wp.array(dtype=wp.float32),
    elas_A_indices: wp.array(dtype=wp.int32),
    elas_indptr: wp.array2d(dtype=wp.int32),
    elas_A_data: wp.array(dtype=wp.float32),
    h: wp.float32,
    P_diag: wp.array(dtype=wp.float32),
):
    con_id = wp.tid()
    
    w = elas_weights[con_id]
    beg = elas_indptr[con_id, 0]
    end = elas_indptr[con_id, 1]
    # Then we have A.data = elas_A[beg:end], A.indices = elas_pids[beg:end]
    
    # Compute diagonal entries of (h^2 Σ w_c A_c^T A_c)
    for k in range(beg, end):
        i = elas_A_indices[k]
        A_i = elas_A_data[k]
        wp.atomic_add(P_diag, i, w * h * h * A_i * A_i)
    

@wp.kernel
def compute_f_elastic(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.float32),
    elas_weights: wp.array(dtype=wp.float32),
    elas_params: wp.array(dtype=wp.float32),
    elas_A_indices: wp.array(dtype=wp.int32),
    elas_A_indptr: wp.array2d(dtype=wp.int32),
    elas_A_data: wp.array(dtype=wp.float32),
    h: wp.float32,
    # b: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
):
    # Compute f = b - R(P) v, where R(P) is the off-diagonal part of P
    # where b = M s + h Σ w_c A_c^T (p_c - A_c q_prev)

    con_id = wp.tid()
    
    w = elas_weights[con_id]
    beg = elas_A_indptr[con_id, 0]
    end = elas_A_indptr[con_id, 1]
    # Then we have A.data = elas_A_data[beg:end], A.indices = elas_A_indices[beg:end]
    
    # Compute (A_c q), (A_c q_prev)
    A_q = wp.vec3(0.0, 0.0, 0.0)
    A_q_prev = wp.vec3(0.0, 0.0, 0.0)
    for i_ind in range(beg, end):  # idx in elas_A_data and elas_A_indices
        i = elas_A_indices[i_ind]  # particle id
        A_i = elas_A_data[i_ind]
        q_i = q[i]
        q_prev_i = q_prev[i]
        A_q += A_i * q_i
        A_q_prev += A_i * q_prev_i

    # Compute p_c
    p = wp.normalize(A_q) * elas_params[con_id]
    
    # Accumulate onto f:
    #     h Σ w_c A_c^T (p_c - A_c q_prev) - R(h^2 Σ w_c A_c^T A_c) v
    for i_ind in range(beg, end):
        i = elas_A_indices[i_ind]
        if masks[i] != 0.0:
            A_i = elas_A_data[i_ind]
            P_off_v_i = wp.vec3(0.0, 0.0, 0.0)
            for j_ind in range(beg, end):
                j = elas_A_indices[j_ind]
                A_j = elas_A_data[j_ind]
                if i != j:
                    P_off_v_i += (w * h * h * A_i * A_j) * qd[j]
            b_upd = (h * w * A_i) * (p - A_q_prev) - P_off_v_i
            wp.atomic_add(f, i, b_upd)


@wp.func
def interpolate_collider_float(
    collider_indices: wp.array2d(dtype=wp.int32),
    collider_weights: wp.array2d(dtype=wp.float32),
    a: wp.array(dtype=wp.float32),
    i: int,
):
    return (
        a[collider_indices[i, 0]] * collider_weights[i, 0]
        + a[collider_indices[i, 1]] * collider_weights[i, 1]
        + a[collider_indices[i, 2]] * collider_weights[i, 2]
    )


@wp.func
def interpolate_collider_vec3(
    collider_indices: wp.array2d(dtype=wp.int32),
    collider_weights: wp.array2d(dtype=wp.float32),
    a: wp.array(dtype=wp.vec3),
    i: int,
):
    return (
        a[collider_indices[i, 0]] * collider_weights[i, 0]
        + a[collider_indices[i, 1]] * collider_weights[i, 1]
        + a[collider_indices[i, 2]] * collider_weights[i, 2]
    )


@wp.func
def distribute_collider_vec3(
    collider_indices: wp.array2d(dtype=wp.int32),
    collider_weights: wp.array2d(dtype=wp.float32),
    a: wp.array(dtype=wp.vec3),
    i: int,
    v: wp.vec3,
):
    wp.atomic_add(a, collider_indices[i, 0], v * collider_weights[i, 0])
    wp.atomic_add(a, collider_indices[i, 1], v * collider_weights[i, 1])
    wp.atomic_add(a, collider_indices[i, 2], v * collider_weights[i, 2])


@wp.func
def distribute_collider_mat33(
    collider_indices: wp.array2d(dtype=wp.int32),
    collider_weights: wp.array2d(dtype=wp.float32),
    a: wp.array(dtype=wp.mat33),
    i: int,
    v: wp.mat33,
):
    wp.atomic_add(a, collider_indices[i, 0], v * collider_weights[i, 0])
    wp.atomic_add(a, collider_indices[i, 1], v * collider_weights[i, 1])
    wp.atomic_add(a, collider_indices[i, 2], v * collider_weights[i, 2])


@wp.kernel
def compute_collider_q(
    collider_indices: wp.array2d(dtype=wp.int32),
    collider_weights: wp.array2d(dtype=wp.float32),
    q: wp.array(dtype=wp.vec3),
    collider_q: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    collider_q[i] = interpolate_collider_vec3(collider_indices, collider_weights, q, i)


@wp.func
def plane_sdf(x: wp.vec3, width: float, length: float):  # width along y, length along z
    # if width > 0.0 and length > 0.0:
    #     d = wp.max(wp.abs(x[1]) - width, wp.abs(x[2]) - length)
    #     return wp.max(d, wp.abs(x[0]))
    return x[0]


@wp.func
def sphere_sdf(x: wp.vec3, r: float):
    return wp.length(x) - r

@wp.func
def box_sdf(x: wp.vec3, scale: wp.vec3):
    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    qx = abs(x[0]) - scale[0]
    qy = abs(x[1]) - scale[1]
    qz = abs(x[2]) - scale[2]

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)

@wp.func
def box_sdf_grad(p: wp.vec3, scale: wp.vec3):
    qx = abs(p[0]) - scale[0]
    qy = abs(p[1]) - scale[1]
    qz = abs(p[2]) - scale[2]

    # exterior case
    if qx > 0.0 or qy > 0.0 or qz > 0.0:
        x = wp.clamp(p[0], -scale[0], scale[0])
        y = wp.clamp(p[1], -scale[1], scale[1])
        z = wp.clamp(p[2], -scale[2], scale[2])

        return wp.normalize(p - wp.vec3(x, y, z))

    sx = wp.sign(p[0])
    sy = wp.sign(p[1])
    sz = wp.sign(p[2])

    # x projection
    if qx > qy and qx > qz or qy == 0.0 and qz == 0.0:
        return wp.vec3(sx, 0.0, 0.0)

    # y projection
    if qy > qx and qy > qz or qx == 0.0 and qz == 0.0:
        return wp.vec3(0.0, sy, 0.0)

    # z projection
    return wp.vec3(0.0, 0.0, sz)


@wp.kernel
def compute_collision_point_body(
    n_shapes: int,
    collider_indices: wp.array2d(dtype=wp.int32),
    collider_weights: wp.array2d(dtype=wp.float32),
    masses: wp.array(dtype=wp.float32),
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    particle_fric: wp.array(dtype=wp.float32),
    body_shape_types: wp.array(dtype=wp.int32),
    shape_scales: wp.array(dtype=wp.vec3),
    shape_volumes: wp.array(dtype=wp.uint64),
    body_q: wp.array(dtype=wp.transform),
    # body_qd: wp.array(dtype=wp.spatial_vector),  # twist
    body_q_prev: wp.array(dtype=wp.transform),
    shape_body_ids: wp.array(dtype=wp.int32),
    shape2cm: wp.array(dtype=wp.transform), # X_bs, shape to body
    shape_fric: wp.array(dtype=wp.float32),
    w: wp.float32,
    collision_sphere_radius: wp.float32,
    h: wp.float32,
    collision_margin: wp.float32,
    f: wp.array(dtype=wp.vec3),
    P_diag: wp.array(dtype=wp.float32),
    r: wp.array(dtype=wp.vec3),
    collision_P: wp.array(dtype=wp.mat33),
    body_f_ext: wp.array(dtype=wp.spatial_vector),
):
    particle_i = wp.tid()

    contact_cnt = float(0.0)  # number of contact forces
    fric_sum = wp.vec3(0.0, 0.0, 0.0)
    f_coll_sum = wp.vec3(0.0, 0.0, 0.0)
    P_coll_sum = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for shape_i in range(n_shapes):
        body_i = shape_body_ids[shape_i]

        particle_mu = interpolate_collider_float(collider_indices, collider_weights, particle_fric, particle_i)
        shape_mu = shape_fric[shape_i]
        # mu = wp.sqrt(particle_mu * shape_mu)  # geometric mean
        mu = (particle_mu + shape_mu) / 2.0  # arithmetic mean

        scale = shape_scales[shape_i]

        par_x_w = interpolate_collider_vec3(collider_indices, collider_weights, q, particle_i)
        # particle position in world frame
        par_v_w = interpolate_collider_vec3(collider_indices, collider_weights, qd, particle_i)
        par_x_w_prev = interpolate_collider_vec3(collider_indices, collider_weights, q_prev, particle_i)

        X_wb = body_q[body_i]  # body CM to world
        X_wb_prev = body_q_prev[body_i]
        # X_bw = wp.transform_inverse(X_wb)
        X_bs = shape2cm[shape_i]  # shape to body CM
        X_ws = wp.transform_multiply(X_wb, X_bs)
        X_ws_prev = wp.transform_multiply(X_wb_prev, X_bs)
        X_sw = wp.transform_inverse(X_ws)
        par_x_s = wp.transform_point(X_sw, par_x_w)  # particle position in shape frame
        par_x_b = wp.transform_point(X_bs, par_x_s)  # particle position in body frame
        # V_w = body_qd[shape_i]  # CM twist in world frame
        # w_w = wp.spatial_top(V_w)
        # v_w = wp.spatial_bottom(V_w)
        # body_v_w = v_w + wp.cross(w_w, par_x_w)  # body velocity at particle position
        body_x_w_prev = wp.transform_point(X_ws_prev, par_x_s)  
        # the contact point on the body in world frame at the previous time step
        body_v_w = (par_x_w - body_x_w_prev) / h
        # the (linearized) velocity of the contact point on the body in world frame

        d = 1.0e6
        n_s = wp.vec3(0.0, 0.0, 0.0)

        if body_shape_types[shape_i] == ShapeTypes.GEO_PLANE:  # plane
            d = plane_sdf(par_x_s, scale[1], scale[2])
            n_s = wp.vec3(1.0, 0.0, 0.0)
        elif body_shape_types[shape_i] == ShapeTypes.GEO_SPHERE:  # sphere
            d = sphere_sdf(par_x_s, scale[0])
            n_s = wp.normalize(par_x_s)
        elif body_shape_types[shape_i] == ShapeTypes.GEO_BOX:
            d = box_sdf(par_x_s, scale)
            n_s = box_sdf_grad(par_x_s, scale)
        elif body_shape_types[shape_i] == ShapeTypes.GEO_SDF:  # SDF
            # NOTE: assume scale[0] == scale[1] == scale[2]
            volume = shape_volumes[shape_i]
            par_x_s_index = wp.volume_world_to_index(volume, wp.cw_div(par_x_s, scale))
            nn = wp.vec3(0.0, 0.0, 0.0)
            d = wp.volume_sample_grad_f(volume, par_x_s_index, wp.Volume.LINEAR, nn)
            d = d * scale[0]

            # wp.printf("d: %f, scale[0] = %f, nn = (%f %f %f)\n", d, scale[0], nn[0], nn[1], nn[2])
            n_s = wp.normalize(nn)

        d = d - collision_sphere_radius
        n_w = wp.quat_rotate(wp.transform_get_rotation(X_ws), n_s)

        if d < collision_margin:
            A_coll = wp.outer(n_w, n_w)
            p_coll = par_x_w + (collision_margin - d) * n_w
            b_coll = (h * w) * A_coll * (p_coll - par_x_w_prev)
            P_coll = (h * h * w) * A_coll

            f_i = interpolate_collider_vec3(collider_indices, collider_weights, f, particle_i)  # = (b - R(P) v)_i, depends only on elastic forces, f_wo_contact_i in prototype
            f_coll_i = b_coll - mat33_off_diag(P_coll) * par_v_w  # f_contact_offdiag_i in prototype
            support_i = b_coll - P_coll * par_v_w  # for computing max friction force
            fric_i = wp.vec3(0.0, 0.0, 0.0)
            
            # support_i = (collision_margin - d) * n_w * w * h

            # Compute the local basis {n, beta1, beta2}
            # Avoid numerical instability when n is close to (1, 0, 0)
            beta1 = wp.vec3(1.0, 0.0, 0.0)
            if wp.abs(n_w[0]) > 0.9:
                beta1 = wp.vec3(0.0, 1.0, 0.0)
            beta1 = wp.normalize(beta1 - wp.dot(n_w, beta1) * n_w)
            beta2 = wp.cross(n_w, beta1)

            # Solve r = a1 * beta1 + a2 * beta2 s.t.
            #     [beta1, beta2]^T [a1, a2]^T (D^-1 f - v_b + D^-1 r) = [0, 0]^T

            D = (
                wp.vec3(interpolate_collider_float(collider_indices, collider_weights, P_diag, particle_i))
                + wp.get_diag(P_coll)
            )
            D_inv_f_minus_v_b = wp.cw_div(f_i + f_coll_i, D) - body_v_w
            eq_b = -1.0 * wp.vec2(
                wp.dot(beta1, D_inv_f_minus_v_b),
                wp.dot(beta2, D_inv_f_minus_v_b),
            )
            eq_A = wp.mat22(
                wp.dot(beta1, wp.cw_div(beta1, D)),
                wp.dot(beta1, wp.cw_div(beta2, D)),
                wp.dot(beta2, wp.cw_div(beta1, D)),
                wp.dot(beta2, wp.cw_div(beta2, D)),
            )
            a = wp.inverse(eq_A) * eq_b
            fric_i = a[0] * beta1 + a[1] * beta2
            support_i_norm = wp.max(wp.dot(support_i, n_w), 0.0)
            fric_i = wp.normalize(fric_i) * wp.min(wp.length(fric_i), mu * support_i_norm)

            contact_cnt += 1.0
            fric_sum += fric_i
            f_coll_sum += f_coll_i
            P_coll_sum += P_coll

            # Compute torque on the body from support_i and fric_i
            # and save it to body_f_ext
            # tau = r x (-support - fric)
            # r = par_x_w - body_cm_w
            
            # wp.printf("support_i = (%f %f %f), fric_i = (%f %f %f)\n", support_i[0], support_i[1], support_i[2], fric_i[0], fric_i[1], fric_i[2])
            
            body_cm_w = wp.transform_get_translation(X_wb)
            r_w = par_x_w - body_cm_w
            # wp.printf("r_w = (%f %f %f)\n", r_w[0], r_w[1], r_w[2])
            body_f = (-support_i - fric_i) / h
            # body_f = (-support_i) / h
            body_torque = wp.cross(r_w, body_f)
        
            # if particle_i == 1234:
            #     wp.printf("mass = %f, body_f = (%f %f %f), par_v_w = (%f %f %f), par_x_w = (%f %f %f)\n", masses[particle_i], body_f[0], body_f[1], body_f[2], par_v_w[0], par_v_w[1], par_v_w[2], par_x_w[0], par_x_w[1], par_x_w[2])
            
            wp.atomic_add(body_f_ext, body_i, wp.spatial_vector(body_torque, body_f))

    if contact_cnt > 0.0:
        r_upd = fric_sum / contact_cnt + f_coll_sum

        distribute_collider_vec3(collider_indices, collider_weights, r, particle_i, r_upd)
        distribute_collider_mat33(collider_indices, collider_weights, collision_P, particle_i, P_coll_sum)


@wp.kernel
def compute_collision_point_point(
    grid: wp.uint64,
    collider_indices: wp.array2d(dtype=wp.int32),
    collider_weights: wp.array2d(dtype=wp.float32),
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    q_prev: wp.array(dtype=wp.vec3),
    qd_prev: wp.array(dtype=wp.vec3),
    q_rest: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    particle_component_ids: wp.array(dtype=wp.int32),
    particle_fric: wp.array(dtype=wp.float32),
    w: wp.float32,  # collision "spring" stiffness
    collision_sphere_radius: wp.float32,
    h: wp.float32,
    collision_margin: wp.float32,
    f: wp.array(dtype=wp.vec3),
    P_diag: wp.array(dtype=wp.float32),
    r: wp.array(dtype=wp.vec3),
    collision_P: wp.array(dtype=wp.mat33),
): 
    i0 = wp.tid()  # collider id; not particle id

    x0 = interpolate_collider_vec3(collider_indices, collider_weights, q, i0)
    x0_rest = interpolate_collider_vec3(collider_indices, collider_weights, q_rest, i0)
    v0 = interpolate_collider_vec3(collider_indices, collider_weights, qd, i0)
    x0_prev = interpolate_collider_vec3(collider_indices, collider_weights, q_prev, i0)
    m0 = interpolate_collider_float(collider_indices, collider_weights, masses, i0)
    cid0 = particle_component_ids[collider_indices[i0, 0]]
    mu0 = interpolate_collider_float(collider_indices, collider_weights, particle_fric, i0)

    query = wp.hash_grid_query(grid, x0, collision_margin + 2.0 * collision_sphere_radius)
    i1 = int(0)
    contact_cnt = float(0.0)  # number of contact forces
    fric_0_sum = wp.vec3(0.0, 0.0, 0.0)
    f0_coll_sum = wp.vec3(0.0, 0.0, 0.0)
    P0_coll_sum = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    while wp.hash_grid_query_next(query, i1):
        x1 = interpolate_collider_vec3(collider_indices, collider_weights, q, i1)
        x1_rest = interpolate_collider_vec3(collider_indices, collider_weights, q_rest, i1)
        n = wp.normalize(x0 - x1)
        d = wp.length(x0 - x1) - 2.0 * collision_sphere_radius
        d_rest = wp.length(x0_rest - x1_rest) - 2.0 * collision_sphere_radius
        cid1 = particle_component_ids[collider_indices[i1, 0]]
        if d < collision_margin and (cid0 != cid1 or d_rest >= 2.0 * collision_margin):
            v1 = interpolate_collider_vec3(collider_indices, collider_weights, qd, i1)
            x1_prev = interpolate_collider_vec3(collider_indices, collider_weights, q_prev, i1)
            m1 = interpolate_collider_float(collider_indices, collider_weights, masses, i1)
            mu1 = interpolate_collider_float(collider_indices, collider_weights, particle_fric, i1)
            # mu = wp.sqrt(mu0 * mu1)  # geometric mean
            mu = (mu0 + mu1) / 2.0  # arithmetic mean

            delta_x = (collision_margin - d) * n
            s0 = m1 / (m0 + m1)
            s1 = m0 / (m0 + m1)
            w0 = w / s0
            w1 = w / s1
            delta_x0 = s0 * delta_x
            delta_x1 = -s1 * delta_x
            p0_coll = x0 + delta_x0
            p1_coll = x1 + delta_x1
            A_coll = wp.outer(n, n)
            b0_coll = (h * w0) * A_coll * (p0_coll - x0_prev)
            # b1_coll = (h * w1) * A_coll * (p1_coll - x1_prev)
            P0_coll = (h * h * w0) * A_coll
            P1_coll = (h * h * w1) * A_coll

            P0_coll_sum += P0_coll

            f0 = interpolate_collider_vec3(collider_indices, collider_weights, f, i0)
            f1 = interpolate_collider_vec3(collider_indices, collider_weights, f, i1)
            f0_coll = b0_coll - mat33_off_diag(P0_coll) * v0
            # f1_coll = b1_coll - mat33_off_diag(P1_coll) * v1
            support_0 = h * w * delta_x
            support_1 = -support_0

            fric_0 = wp.vec3(0.0, 0.0, 0.0)
            fric_1 = wp.vec3(0.0, 0.0, 0.0)

            D0 = (
                wp.vec3(interpolate_collider_float(collider_indices, collider_weights, P_diag, i0))
                + wp.get_diag(P0_coll)
            )
            D1 = (
                wp.vec3(interpolate_collider_float(collider_indices, collider_weights, P_diag, i1))
                + wp.get_diag(P1_coll)
            )
            sum01_D = D0 + D1
            delta_f = wp.cw_mul(D1, f0) - wp.cw_mul(D0, f1)
            delta_f_T = delta_f - wp.dot(delta_f, n) * n
            sum01_D_support_norm = wp.length(wp.cw_mul(sum01_D, support_0))
            sum01_D_r_T = -wp.normalize(delta_f_T) * wp.min(wp.length(delta_f_T), mu * sum01_D_support_norm)
            fric_0 = wp.cw_div(sum01_D_r_T, sum01_D)
            fric_1 = -fric_0

            r0 = f0_coll + fric_0
            # r1 = f_col_1 + fric_1
            fric_0_sum += fric_0
            f0_coll_sum += f0_coll
            contact_cnt += 1.0
    
    if contact_cnt > 0.0:
        # wp.atomic_add(r, i0, (fric_0_sum + f0_coll_sum) / contact_cnt)
        # wp.atomic_add(collision_P, i0, P0_coll_sum / contact_cnt)

        r0_upd = fric_0_sum / contact_cnt + f0_coll_sum
        distribute_collider_vec3(collider_indices, collider_weights, r, i0, r0_upd)
        distribute_collider_mat33(collider_indices, collider_weights, collision_P, i0, P0_coll_sum)

        # wp.atomic_add(r, i0, fric_0_sum + f0_coll_sum)
        # wp.atomic_add(collision_P, i0, P0_coll_sum)


@wp.func
def mat33_off_diag(a: wp.mat33):
    return a - wp.diag(wp.get_diag(a))


@wp.kernel
def jacobi_step_v(
    f: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    P_diag: wp.array(dtype=wp.float32),
    collision_P: wp.array(dtype=wp.mat33),
    masks: wp.array(dtype=wp.float32),
    qd_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if masks[i] != 0.0:
        D_i = wp.vec3(P_diag[i]) + wp.get_diag(collision_P[i])
        qd_out[i] = wp.cw_div(f[i] + r[i], D_i)


@wp.kernel
def compute_q_from_qd(
    q_prev: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    h: wp.float32,
    q: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    q[i] = q_prev[i] + h * qd[i]


@wp.kernel
def chebyshev_update(
    qd_prev_iter: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    qd_next_iter: wp.array(dtype=wp.vec3),
    omega: wp.float32,
    gamma: wp.float32,
): 
    i = wp.tid()
    qd_prev_i = qd_prev_iter[i]
    q_i = qd[i]
    qd_next_i = qd_next_iter[i]
    qd[i] = omega * (gamma * (qd_next_i - q_i) + q_i - qd_prev_i) + qd_prev_i
    qd_prev_iter[i] = q_i

