from dataclasses import dataclass

from perlin_numpy import generate_fractal_noise_2d
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

    def get_terrain_height(self, pos) -> float:
        # Get the heightfield (shape: (rows, cols))
        heightfield = self.heights
        rows, cols = heightfield.shape

        terrain_width = self.grid_width
        terrain_depth = self.grid_depth

        # Map world x,z (pos.x, pos.y) to grid indices (floating point)
        fx = (pos.x + terrain_width / 2) / terrain_width * (cols - 1)
        fy = (pos.y + terrain_depth / 2) / terrain_depth * (rows - 1)

        # Clamp the floating point indices to be within valid range
        fx = max(0, min(cols - 1, fx))
        fy = max(0, min(rows - 1, fy))

        # Get the integer positions for interpolation
        x0 = int(fx)
        y0 = int(fy)
        x1 = min(x0 + 1, cols - 1)
        y1 = min(y0 + 1, rows - 1)

        # Compute the fractional part
        tx = fx - x0
        ty = fy - y0

        # Retrieve the heights from the four surrounding grid points
        h00 = heightfield[y0, x0]
        h10 = heightfield[y0, x1]
        h01 = heightfield[y1, x0]
        h11 = heightfield[y1, x1]

        # Perform bilinear interpolation
        height = (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 \
                + (1 - tx) * ty * h01 + tx * ty * h11
        return height


def blank_terrain(
    segments: int,
    grid_width: float,
    grid_depth: float,
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
            idx = i * (segments + 1) + j
            vertices["in_position"][idx] = (x, 0, z)
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


def make_terrain(
    segments: int,
    grid_width: float,
    grid_depth: float,
    height_multiplier: float,
    noise_scale: float,
) -> Mesh:
    """Create a grid of (segments+1) x (segments+1) vertices."""
    mesh = blank_terrain(segments, grid_width, grid_depth)
    heights = mesh.heights

    # Generate 2D Perlin fractal noise for the heights
    res_x = max(1, int(1/noise_scale))
    res_y = max(1, int(1/noise_scale))
    noise = generate_fractal_noise_2d(
        (segments + 1, segments + 1),
        (res_y, res_x),
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        rng=np.random.default_rng(42)
    )
    heights[:, :] = noise
    heights *= height_multiplier
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

    # Compute normals using finite (central) differences with numpy operations
    heights = mesh.heights

    # Create arrays for neighboring heights with proper edge handling
    heights_left = np.roll(heights, 1, axis=1)
    heights_left[:, 0] = heights[:, 0]  # Edge case

    heights_right = np.roll(heights, -1, axis=1)
    heights_right[:, -1] = heights[:, -1]  # Edge case

    heights_down = np.roll(heights, 1, axis=0)
    heights_down[0, :] = heights[0, :]  # Edge case

    heights_up = np.roll(heights, -1, axis=0)
    heights_up[-1, :] = heights[-1, :]  # Edge case

    # Compute derivatives
    dX = (heights_right - heights_left) / (2 * dx)
    dZ = (heights_up - heights_down) / (2 * dz)

    # Create normal vectors
    normals = np.zeros((segments + 1, segments + 1, 3), dtype=np.float32)
    normals[:, :, 0] = -dX
    normals[:, :, 1] = 1.0
    normals[:, :, 2] = -dZ

    # Normalize normals
    norm = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    normals = normals / norm

    # Assign to vertex normals
    vertices["in_normal"] = normals.reshape(-1, 3)

    return mesh
