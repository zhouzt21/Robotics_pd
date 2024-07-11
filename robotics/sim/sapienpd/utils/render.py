import warp as wp


@wp.kernel
def copy_positions_to_render(
    dst_vertices: wp.array2d(dtype=wp.float32),
    src_positions: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    dst_vertices[i, 0] = src_positions[i][0]
    dst_vertices[i, 1] = src_positions[i][1]
    dst_vertices[i, 2] = src_positions[i][2]

