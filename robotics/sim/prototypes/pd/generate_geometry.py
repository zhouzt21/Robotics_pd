import igl
import numpy as np
import scipy.sparse as sp


cloth_res = (11, 11)
cloth_shape = (1.0, 1.0)

grid_vertices_2d = q_rest_arr2d = np.stack(
    np.meshgrid(
        np.linspace(0, cloth_shape[0], cloth_res[0], dtype=np.float32),
        np.linspace(0, cloth_shape[1], cloth_res[1], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        indexing="ij",
    ),
    axis=-1,
)
grid_vertices_id_2d = np.arange(cloth_res[0] * cloth_res[1], dtype=np.int32).reshape(
    cloth_res
)

# Add one vertex at the center of each square cell
cell_shape = (cloth_shape[0] / (cloth_res[0] - 1), cloth_shape[1] / (cloth_res[1] - 1), 0.)
cell_centers_2d = (
    grid_vertices_2d[:-1, :-1] + np.array(cell_shape, dtype=np.float32) / 2.0
)
cell_centers_id_2d = (
    np.arange((cloth_res[0] - 1) * (cloth_res[1] - 1), dtype=np.int32).reshape(
        (cloth_res[0] - 1, cloth_res[1] - 1)
    )
    + cloth_res[0] * cloth_res[1]
)

# complete vertices
vertices = np.concatenate(
    [
        grid_vertices_2d.reshape(-1, 3),
        cell_centers_2d.reshape(-1, 3),
    ],
    axis=0,
)

# Create 4 triangles for each square cell
#
# g[i, j] ---  g[i, j+1]
#    | \   0   /  |
#    |1  c[i, j] 3|
#    | /   2   \  |
# g[i+1, j] -- g[i+1, j+1]
#
faces = []
for i in range(cloth_res[0] - 1):
    for j in range(cloth_res[1] - 1):
        faces.append(
            (
                grid_vertices_id_2d[i, j],
                grid_vertices_id_2d[i, j + 1],
                cell_centers_id_2d[i, j],
            )
        )
        faces.append(
            (
                grid_vertices_id_2d[i, j],
                cell_centers_id_2d[i, j],
                grid_vertices_id_2d[i + 1, j],
            )
        )
        faces.append(
            (
                grid_vertices_id_2d[i + 1, j],
                cell_centers_id_2d[i, j],
                grid_vertices_id_2d[i + 1, j + 1],
            )
        )
        faces.append(
            (
                cell_centers_id_2d[i, j],
                grid_vertices_id_2d[i, j + 1],
                grid_vertices_id_2d[i + 1, j + 1],
            )
        )
faces = np.array(faces, dtype=np.int32)

igl.write_triangle_mesh(f"cloth_{cloth_res[0]}.obj", vertices, faces)


boundary = igl.boundary_facets(faces)
boundary_vertices = np.unique(boundary)
non_boundary_vertices = np.setdiff1d(np.arange(vertices.shape[0]), boundary_vertices)

# Compute the cotangent weights
cotmatrix = igl.cotmatrix(vertices, faces)
massmatrix = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_VORONOI)
voronoi_areas = massmatrix.diagonal()

# compute the mean curvatures
# HN = -(cotmatrix.dot(vertices) / voronoi_areas[:, None])
HN = -(cotmatrix / voronoi_areas[:, None]).dot(vertices)
curvature = np.linalg.norm(HN, axis=1)
non_boundary_curvature = curvature[non_boundary_vertices]
# print(non_boundary_curvature.max())
# print(curvature)

