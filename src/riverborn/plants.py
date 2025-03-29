import math
import random
import glm
from .scene import Material, Scene, Model
from .terrain import Mesh


class PlantGrid:
    def __init__(self, scene: Scene, terrain: Mesh):
        self.scene = scene
        self.terrain = terrain
        self.plant_grid = {}
        self.plant_material = Material(
            double_sided=True,
            translucent=True,
            transmissivity=0.3,
            receive_shadows=True,  # Plants receive shadows
            cast_shadows=True,     # Plants cast shadows
            alpha_test=True
        )

    def setup(self):
        """
        Setup the plant grid by loading models and placing them on the terrain.
        """
        # Define plant material properties
        # Plants are double-sided and have some translucency
        self.little_plants = [
            self.scene.load_wavefront(fname, material=self.plant_material, capacity=50)
            for fname in [
                "plant01_t.obj",
                "plant02_t.obj",
                "plant03_t.obj",
                "plant04_t.obj",
                "plant05_t.obj",
                "plant06_t.obj",
                "plant07_t.obj",
                "plant08_t.obj",
            ]
        ]

        self.small_plants = [
            self.scene.load_wavefront(fname, material=self.plant_material, capacity=50)
            for fname in [
                "plant09_t.obj",
                "plant10_t.obj",
                "plant11_t.obj",
                "plant12_t.obj",
            ]
        ]
        self.big_plants = [
            self.scene.load_wavefront(fname, material=self.plant_material, capacity=50)
            for fname in [
                "fern.obj",
                "plant13.obj",
                "plant15.obj",
                "plant16.obj",
            ]
        ]
        self.place_plants(self.big_plants, 5, fraction=0.05)
        self.place_plants(self.big_plants, 3, fraction=0.1)
        self.place_plants(self.little_plants, 1)

    def place_plants(self, models: list[Model], size: int, fraction: float = 1.0):
        """Place plants and store their instances in the grid.

        Plants take up a grid of size x size centered around their position.

        """
        # Plant distribution parameters
        terrain_width = 200   # Width of terrain in world units
        terrain_depth = 200   # Depth of terrain in world units
        plant_spacing = 1    # Distance between plants
        water_level = 1.0     # Y position of water surface

        # Store plant instances for future updates

        heights = self.terrain.heights
        h_rows, h_cols = heights.shape

        # Calculate grid dimensions to cover the entire terrain
        grid_size_x = int(terrain_width / plant_spacing)
        grid_size_z = int(terrain_depth / plant_spacing)

        # Place plants in a grid across the entire terrain
        for i in range(-grid_size_x//2, grid_size_x//2):
            for j in range(-grid_size_z//2, grid_size_z//2):
                # Skip if the grid cell is already occupied
                if (i, j) in self.plant_grid:
                    continue

                rng = random.Random(i + 123907 * j)
                # Randomly skip some plants based on the fraction
                if rng.random() > fraction:
                    continue

                # Calculate world position with random offset
                world_x = i * plant_spacing + rng.uniform(-plant_spacing/4, plant_spacing/4)
                world_z = j * plant_spacing + rng.uniform(-plant_spacing/4, plant_spacing/4)

                # Convert world coordinates to normalized coordinates (0-1)
                norm_x = (world_x + terrain_width/2) / terrain_width
                norm_z = (world_z + terrain_depth/2) / terrain_depth

                # Skip if out of bounds
                if not (0 <= norm_x <= 1 and 0 <= norm_z <= 1):
                    continue

                # Perform bilinear filtering to get an accurate height estimate
                x0 = math.floor(norm_x * (h_cols - 1))
                x1 = min(x0 + 1, h_cols - 1)
                z0 = math.floor(norm_z * (h_rows - 1))
                z1 = min(z0 + 1, h_rows - 1)

                sx = (norm_x * (h_cols - 1)) - x0
                sz = (norm_z * (h_rows - 1)) - z0

                h00 = heights[z0, x0]
                h10 = heights[z0, x1]
                h01 = heights[z1, x0]
                h11 = heights[z1, x1]

                height = (1 - sx) * (1 - sz) * h00 + sx * (1 - sz) * h10 + (1 - sx) * sz * h01 + sx * sz * h11

                # Only place plants above water level
                if height > water_level:
                    # Set plant position
                    plant_pos = glm.vec3(world_x, height, world_z)
                else:
                    # Skip if below water level
                    continue

                inst = self.scene.add(rng.choice(models))
                # Random position on the terrain
                inst.pos = plant_pos
                inst.rotate(rng.uniform(-0.3, 0.3), glm.vec3(1, 0, 0))
                inst.rotate(rng.uniform(-0.3, 0.3), glm.vec3(0, 0, 1))
                inst.rotate(rng.uniform(0, 2 * math.pi), glm.vec3(0, 1, 0))
                inst.scale = glm.vec3(rng.uniform(0.05, 0.1))
                inst.update()

                for x in range(-size // 2, size // 2 + 1):
                    for z in range(-size // 2, size // 2 + 1):
                        # Store the instance in the grid
                        self.plant_grid[(i + x, j + z)] = inst

