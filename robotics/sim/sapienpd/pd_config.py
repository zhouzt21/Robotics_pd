import numpy as np
import warp as wp


class PDConfig:
    def __init__(self) -> None:
        ################ Memory config ################
        self.max_scenes = 1 << 10
        self.max_bodies = 1 << 10
        self.max_body_shapes = 1 << 10
        self.max_components = 1 << 20
        self.max_particles = 1 << 20
        self.max_constraints = 1 << 20
        self.max_constraint_sizes_sum = 1 << 20  # set to >= 4*n_edges + n_vertices
        self.max_colliders = 1 << 20  # set to >= n_vertices + 3^(depth - 1) * n_faces
        """ Each edge appears twice in spring constraints and twice in bending constraints;
            each vertex appears once in the bending constraint about itself. """
        
        ################ Solver config ################
        self.time_step = 2e-3
        self.n_pd_iters = 20

        # continuous collision detection
        self.ccd_flag = False
        self.ccd_every = 20

        # use cuda graph to speed up kernel launch
        self.cuda_graph_flag = False

        # use chebyshev method to speed up convergence
        self.chebyshev_flag = True
        self.chebyshev_rho = 0.99
        self.chebyshev_gamma = 0.7
        self.chebyshev_warmup_iters = 5

        # use hash grid to speed up collision detection
        self.hash_grid_flag = True
        self.hash_grid_dim = 128

        ################ Physics config ################
        self.gravity = np.array([0, 0, -9.8], dtype=np.float32)
        self.collision_margin = 1e-3
        self.collision_weight = 5e3
        self.collision_sphere_radius = 8e-3
        self.max_velocity = 1.0  # estimated max velocity for collision detection, TODO: add clamping
        self.damping = 0.0  # damping energy: 0.5 * damping * v^2

        ################ Debug config ################
        self.debug_flag = False
