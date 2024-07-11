import numpy as np


class Foo:
    def __init__(self, collider_iterpolation_depth: int = 1):
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        self.n_vertices = len(self.vertices)
        self.n_faces = len(self.faces)

        collider_vertices = self.vertices.copy()
        # collider_faces = self.faces.copy()
        collider_inter_indices = np.arange(len(self.vertices))[:, None].repeat(3, axis=1)  # [N, 3]
        collider_inter_weights = np.array([[1.0, 0.0, 0.0]], dtype=np.float32).repeat(len(self.vertices), axis=0)  # [N, 3]
        n_colliders = len(collider_vertices)
        last_faces = self.faces.copy()
        for dep in range(collider_iterpolation_depth):
            new_vertices = np.mean(collider_vertices[last_faces], axis=1)  # [LF, 3]
            # Subdivide each face in the last_faces
            # First get the edges
            new_edges = last_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)  # [LF*3, 2]
            # Then append the new vertices
            new_indices_x3 = np.repeat(np.arange(len(new_vertices)), 3, axis=0)[:, None] + n_colliders # [LF*3, 1]
            
            # Compute new interpolation indices
            if dep == 0:
                # Interpolate from the original vertices
                new_collider_inter_indices = last_faces
                new_collider_inter_weights = (np.array([[1.0, 1.0, 1.0]], dtype=np.float32) / 3.0).repeat(len(last_faces), axis=0)
            else:
                # Interpolate from the last interpolation
                new_collider_inter_indices = collider_inter_indices[last_faces[:, 2]]  # [LF, 3], the last vertex cannot be an original vertex
                new_collider_inter_weights = np.zeros((len(last_faces), 3), dtype=np.float32)
                for i in range(3):
                    inter_indices_i = new_collider_inter_indices[:, i]  # [LF]
                    for j in range(3):
                        last_faces_j = last_faces[:, j]  # [LF]
                        mask = collider_inter_indices[last_faces_j] == inter_indices_i[:, None]  # [LF, 3]
                        weights = np.sum(collider_inter_weights[last_faces_j] * mask, axis=1)  # [LF]
                        new_collider_inter_weights[:, i] += weights / 3.0
                
            # Append the new interpolation indices
            collider_inter_indices = np.vstack([collider_inter_indices, new_collider_inter_indices])
            collider_inter_weights = np.vstack([collider_inter_weights, new_collider_inter_weights])  # [N, 3]

            last_faces = np.hstack([new_edges, new_indices_x3]).reshape(-1, 3)  # [LF*3, 3]
            # collider_faces = np.vstack([collider_faces, last_faces])
            collider_vertices = np.vstack([collider_vertices, new_vertices])
            n_colliders += len(new_vertices)

        self.collider_vertices = collider_vertices
        # self.collider_faces = collider_faces
        self.collider_inter_indices = collider_inter_indices
        self.collider_inter_weights = collider_inter_weights

foo = Foo(collider_iterpolation_depth=2)
print(foo.collider_vertices)
# print(foo.collider_faces)
print(foo.collider_inter_indices)  # [N, 3]
print(foo.collider_inter_weights)  # [N, 3]

# Verify interpolation weights
vertices_verify = foo.collider_inter_weights[:, :, None] * foo.collider_vertices[foo.collider_inter_indices]  # [N, 3(triangle), 3(dim)]
vertices_verify = np.sum(vertices_verify, axis=1)  # [N, 3]
assert np.allclose(vertices_verify, foo.collider_vertices)
