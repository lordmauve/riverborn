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
        self.plant_instances = set()  # Track all instances for easier management
        self.plant_material = Material(
            double_sided=True,
            translucent=True,
            transmissivity=0.3,
            receive_shadows=True,  # Plants receive shadows
            cast_shadows=True,     # Plants cast shadows
            alpha_test=True
        )
        # Plant distribution parameters
        self.terrain_width = 200   # Width of terrain in world units
        self.terrain_depth = 200   # Depth of terrain in world units
        self.plant_spacing = 1     # Distance between plants
        self.water_level = 1.0     # Y position of water surface
        self.little_plants = []
        self.small_plants = []
        self.big_plants = []

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
        # Place plants across the entire terrain
        self.place_plants(self.big_plants, 5, fraction=0.05)
        self.place_plants(self.big_plants, 3, fraction=0.1)
        self.place_plants(self.little_plants, 1)

    def world_to_grid(self, world_x, world_z):
        """Convert world coordinates to grid coordinates."""
        grid_x = int(world_x / self.plant_spacing)
        grid_z = int(world_z / self.plant_spacing)
        return grid_x, grid_z

    def grid_to_world(self, grid_x, grid_z):
        """Convert grid coordinates to world coordinates."""
        world_x = grid_x * self.plant_spacing
        world_z = grid_z * self.plant_spacing
        return world_x, world_z

    def delete_plants_in_area(self, world_x, world_z, radius):
        """Delete plants in a square area around the given world position."""
        center_grid_x, center_grid_z = self.world_to_grid(world_x, world_z)
        grid_radius = int(radius / self.plant_spacing) + 1

        # Track instances to delete and grid cells to clear
        instances_to_delete = set()
        grid_cells_to_clear = []

        # Find all instances in the area
        for i in range(center_grid_x - grid_radius, center_grid_x + grid_radius + 1):
            for j in range(center_grid_z - grid_radius, center_grid_z + grid_radius + 1):
                grid_pos = (i, j)
                if grid_pos in self.plant_grid:
                    instance = self.plant_grid[grid_pos]
                    # Only add valid instances to the deletion set
                    if instance is not None and not getattr(instance, '_deleted', False):
                        instances_to_delete.add(instance)
                    # Always clear the cell
                    grid_cells_to_clear.append(grid_pos)

        # Delete the instances
        deleted_count = 0
        for instance in instances_to_delete:
            # Double-check it's not already deleted
            if not getattr(instance, '_deleted', False):
                try:
                    instance.delete()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting instance: {e}")

            # Remove from our tracking set regardless
            self.plant_instances.discard(instance)

        # Clear the grid cells
        for cell in grid_cells_to_clear:
            if cell in self.plant_grid:
                del self.plant_grid[cell]

        #print(f"Deleted {deleted_count} plants from {len(grid_cells_to_clear)} grid cells")
        return grid_radius

    def rebuild_plants_in_area(self, world_x, world_z, radius):
        """Rebuild plants in a square area around the given world position."""
        # First, delete existing plants in the area
        grid_radius = self.delete_plants_in_area(world_x, world_z, radius)
        center_grid_x, center_grid_z = self.world_to_grid(world_x, world_z)

        # Define the grid bounds for rebuilding
        min_grid_x = center_grid_x - grid_radius
        max_grid_x = center_grid_x + grid_radius + 1  # Make inclusive
        min_grid_z = center_grid_z - grid_radius
        max_grid_z = center_grid_z + grid_radius + 1  # Make inclusive

        # Place plants in the area
        if self.big_plants:
            self.place_plants(self.big_plants, 5, fraction=0.05,
                              grid_bounds=(min_grid_x, max_grid_x, min_grid_z, max_grid_z))
            self.place_plants(self.big_plants, 3, fraction=0.1,
                              grid_bounds=(min_grid_x, max_grid_x, min_grid_z, max_grid_z))

        if self.little_plants:
            self.place_plants(self.little_plants, 1,
                              grid_bounds=(min_grid_x, max_grid_x, min_grid_z, max_grid_z))

    def place_plants(self, models: list[Model], size: int, fraction: float = 1.0,
                     grid_bounds=None):
        """Place plants and store their instances in the grid.

        Plants take up a grid of size x size centered around their position.

        Args:
            models: List of plant models to choose from
            size: Size of the plant in grid cells
            fraction: Fraction of grid cells to place plants in
            grid_bounds: Optional tuple (min_x, max_x, min_z, max_z) to limit plant placement
        """
        heights = self.terrain.heights
        h_rows, h_cols = heights.shape

        # Calculate grid dimensions
        if grid_bounds:
            min_grid_x, max_grid_x, min_grid_z, max_grid_z = grid_bounds
        else:
            # Default to covering the entire terrain
            grid_size_x = int(self.terrain_width / self.plant_spacing)
            grid_size_z = int(self.terrain_depth / self.plant_spacing)
            min_grid_x, max_grid_x = -grid_size_x//2, grid_size_x//2
            min_grid_z, max_grid_z = -grid_size_z//2, grid_size_z//2

        # Check for invalid model list
        if not models or len(models) == 0:
            print("Warning: No plant models provided to place_plants")
            return

        plants_placed = 0
        failed_positions = 0

        # Place plants in the specified grid area
        for i in range(min_grid_x, max_grid_x):
            for j in range(min_grid_z, max_grid_z):
                # Skip if the grid cell is already occupied
                if (i, j) in self.plant_grid:
                    continue

                # Use a deterministic seed for random generation
                rng = random.Random(i * 10000 + j)

                # Randomly skip some plants based on the fraction
                if rng.random() > fraction:
                    continue

                # Calculate world position with random offset
                world_x, world_z = self.grid_to_world(i, j)
                world_x += rng.uniform(-self.plant_spacing/4, self.plant_spacing/4)
                world_z += rng.uniform(-self.plant_spacing/4, self.plant_spacing/4)

                # Convert world coordinates to normalized coordinates (0-1)
                norm_x = (world_x + self.terrain_width/2) / self.terrain_width
                norm_z = (world_z + self.terrain_depth/2) / self.terrain_depth

                # Skip if out of bounds
                if not (0 <= norm_x <= 1 and 0 <= norm_z <= 1):
                    failed_positions += 1
                    continue

                # Get height with bilinear filtering
                try:
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
                except Exception as e:
                    print(f"Error calculating height at {norm_x}, {norm_z}: {e}")
                    failed_positions += 1
                    continue

                # Only place plants above water level
                if height <= self.water_level:
                    # Skip if below water level
                    continue

                try:
                    # Set plant position
                    plant_pos = glm.vec3(world_x, height, world_z)

                    # Create the instance
                    inst = self.scene.add(rng.choice(models))

                    # Configure the instance
                    inst.pos = plant_pos
                    inst.rotate(rng.uniform(-0.3, 0.3), glm.vec3(1, 0, 0))
                    inst.rotate(rng.uniform(-0.3, 0.3), glm.vec3(0, 0, 1))
                    inst.rotate(rng.uniform(0, 2 * math.pi), glm.vec3(0, 1, 0))
                    inst.scale = glm.vec3(rng.uniform(0.05, 0.1))
                    inst.update()

                    # Add the instance to our tracking set
                    self.plant_instances.add(inst)
                    plants_placed += 1

                    # Mark grid cells as occupied by this plant
                    occupied_cells = 0
                    for x in range(-size // 2, size // 2 + 1):
                        for z in range(-size // 2, size // 2 + 1):
                            grid_cell = (i + x, j + z)
                            if grid_cell not in self.plant_grid:
                                self.plant_grid[grid_cell] = inst
                                occupied_cells += 1

                    if occupied_cells == 0:
                        print(f"Warning: Plant created at {i},{j} but no grid cells were marked")

                except Exception as e:
                    print(f"Error creating plant at {world_x}, {world_z}: {e}")
                    failed_positions += 1

        if grid_bounds:  # Only print for localized rebuilds
            #print(f"Placed {plants_placed} plants in area {min_grid_x},{min_grid_z} to {max_grid_x},{max_grid_z}")
            if failed_positions > 0:
                print(f"Failed to place plants at {failed_positions} positions")

    def cleanup(self):
        """Clean up deleted instances and validate grid state."""
        # Remove any deleted instances from the tracking set
        to_remove = set()
        for inst in self.plant_instances:
            if getattr(inst, '_deleted', False):
                to_remove.add(inst)

        if to_remove:
            self.plant_instances.difference_update(to_remove)

        # Remove any grid cells pointing to deleted instances
        grid_cleanup = []
        for pos, inst in self.plant_grid.items():
            if inst is None or getattr(inst, '_deleted', False):
                grid_cleanup.append(pos)

        if grid_cleanup:
            for pos in grid_cleanup:
                del self.plant_grid[pos]


