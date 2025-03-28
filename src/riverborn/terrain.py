from dataclasses import dataclass

import noise
import numpy as np


@dataclass
class Mesh:
    segments: int
    grid_width: float
    grid_depth: float
    vertices: np.ndarray
    indices: np.ndarray

    @property
    def heights(self) -> np.ndarray:
        """Get the heights of the vertices."""
        return self.vertices["in_position"][:, 1].reshape(self.segments + 1, self.segments + 1)


def make_terrain(
    segments: int,
    grid_width: float,
    grid_depth: float,
    height_multiplier: float,
    noise_scale: float,
) -> Mesh:
    """Create a grid of (segments+1) x (segments+1) vertices."""
    num_vertices = (segments + 1) * (segments + 1)
    vertices = np.zeros(
        num_vertices,
        dtype=[
            ("in_position", np.float32, 3),
            ("in_normal", np.float32, 3),
            ("in_uv", np.float32, 2),
        ],
    )

    xs = np.linspace(-grid_width / 2, grid_width / 2, segments + 1)
    zs = np.linspace(-grid_depth / 2, grid_depth / 2, segments + 1)
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]

    # First, compute heights using Perlin noise.
    heights = np.zeros((segments + 1, segments + 1), dtype=np.float32)
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            n = noise.pnoise2(
                x * noise_scale,
                z * noise_scale,
                octaves=4,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=1024,
                repeaty=1024,
                base=42,
            )
            h = n * height_multiplier
            heights[i, j] = h
            idx = i * (segments + 1) + j
            vertices["in_position"][idx] = (x, h, z)
            vertices["in_uv"][idx] = (j / segments, i / segments)

    # Build indices for drawing triangles.
    indices = []
    for i in range(segments):
        for j in range(segments):
            top_left = i * (segments + 1) + j
            top_right = top_left + 1
            bottom_left = (i + 1) * (segments + 1) + j
            bottom_right = bottom_left + 1
            indices.extend(
                [
                    top_left,
                    bottom_left,
                    top_right,
                    top_right,
                    bottom_left,
                    bottom_right,
                ]
            )
    indices = np.array(indices, dtype=np.int32)
    mesh = Mesh(
        segments=segments,
        grid_width=grid_width, grid_depth=grid_depth, vertices=vertices, indices=indices
    )
    recompute_normals(mesh)
    return mesh



def recompute_normals(mesh: Mesh) -> Mesh:
    """Recompute normals for the given terrain mesh."""
    segments = mesh.segments
    vertices = mesh.vertices
    grid_width = mesh.grid_width
    grid_depth = mesh.grid_depth

    xs = np.linspace(-grid_width / 2, grid_width / 2, segments + 1)
    zs = np.linspace(-grid_depth / 2, grid_depth / 2, segments + 1)
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]

    # Compute normals using finite (central) differences.
    vertex_normals = vertices["in_normal"].reshape(segments + 1, segments + 1, 3)
    heights = vertices['in_position'][:, 1].reshape(segments + 1, segments + 1)
    for i in range(segments + 1):
        for j in range(segments + 1):
            hL = heights[i, j - 1] if j > 0 else heights[i, j]
            hR = heights[i, j + 1] if j < segments else heights[i, j]
            hD = heights[i - 1, j] if i > 0 else heights[i, j]
            hU = heights[i + 1, j] if i < segments else heights[i, j]
            dX = (hR - hL) / (2 * dx)
            dZ = (hU - hD) / (2 * dz)
            n = np.array([-dX, 1.0, -dZ], dtype=np.float32)
            n /= np.linalg.norm(n)
            vertex_normals[i, j, :] = n
    return mesh
