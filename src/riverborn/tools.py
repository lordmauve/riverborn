import math
import typing

import numpy as np
from pyglm import glm

from .terrain import recompute_normals

if typing.TYPE_CHECKING:
    from .mgl_terrain import WaterApp

TOOLS = []

def register_tool(tool_class):
    """Register a tool class for use in the application."""
    TOOLS.append(tool_class)
    return tool_class


@register_tool
class WaterDisturbTool:
    """Disturb the water surface with mouse input."""
    last_mouse: tuple[float, float] | None = None

    def __init__(self, app: 'WaterApp'):
        self.app = app
        self.last_mouse = None

    def update(self, dt: float):
        pass

    def on_mouse_drag_event(self, x, y, dx, dy):
        # Convert mouse coordinates (window: origin top-left) to texture coordinates (origin bottom-left)
        cur_pos = self.app.screen_to_water(x, y)

        if cur_pos is None:
            self.last_mouse = None
            return

        if self.last_mouse is None:
            self.last_mouse = cur_pos

        # Apply a disturbance by drawing between the last and current mouse positions.
        self.app.water_sim.disturb(self.last_mouse, cur_pos)
        self.last_mouse = cur_pos

    def on_mouse_press_event(self, x, y, button):
        # Record the initial mouse position in texture coordinates.
        self.last_mouse = self.app.screen_to_water(x, y)

    def on_mouse_release_event(self, x, y, button):
        self.last_mouse = None


@register_tool
class RaiseTool:
    """Raise/Lower the terrain surface"""

    def __init__(self, app: 'WaterApp'):
        self.app = app
        self.last_mouse = None
        self.speed = 0

    def update(self, dt: float):
        if self.last_mouse is None:
            return

        model = self.app.terrain_instance.model
        heights = model.mesh.heights

        w, h = heights.shape
        x, y = self.last_mouse

        # numpy distance from pos
        coords_x, coords_y = np.indices((w, h))
        dist = np.sqrt(
            (coords_x - h * y) ** 2 + (coords_y - w * x) ** 2
        )
        if self.speed > 0:
            # Raise terrain
            width = 4
        else:
            # Lower terrain
            width = 2
        heights += self.speed * 0.2 * np.exp(-dist / width)
        recompute_normals(model.mesh)
        model.update_mesh()

        # Convert texture coordinates to world position
        norm_x, norm_z = x, y
        world_x = (norm_x * self.app.terrain_width) - (self.app.terrain_width / 2)
        world_z = (norm_z * self.app.terrain_depth) - (self.app.terrain_depth / 2)
        radius = 8 if self.speed > 0 else 4

        self.app.plants.rebuild_plants_in_area(world_x, world_z, radius)

    def on_mouse_drag_event(self, x, y, dx, dy):
        # Convert mouse coordinates (window: origin top-left) to texture coordinates (origin bottom-left)
        self.last_mouse = self.app.screen_to_water(x, y)

    def on_mouse_press_event(self, x, y, button):
        # Record the initial mouse position in texture coordinates.
        self.last_mouse = self.app.screen_to_water(x, y)
        self.speed = 1 if button == self.app.wnd.mouse_states.left else -1

    def on_mouse_release_event(self, x, y, button):
        self.last_mouse = None


@register_tool
class SmoothTool:
    """Smooth the terrain surface."""

    def __init__(self, app: 'WaterApp'):
        self.app = app
        self.last_mouse = None
        self.speed = 0
        self.target_height = None

    def update(self, dt: float):
        if self.last_mouse is None or self.target_height is None:
            return

        model = self.app.terrain_instance.model
        heights = model.mesh.heights

        w, h = heights.shape
        x, y = self.last_mouse

        # numpy distance from pos
        coords_x, coords_y = np.indices((w, h))
        dist = np.sqrt(
            (coords_x - h * y) ** 2 + (coords_y - w * x) ** 2
        )
        effect = np.exp(-dist / 4) * (1 - 0.2 ** dt)
        heights[:] = heights * (1 - effect) + self.target_height * effect

        recompute_normals(model.mesh)
        model.update_mesh()

        # Convert texture coordinates to world position
        norm_x, norm_z = x, y
        world_x = (norm_x * self.app.terrain_width) - (self.app.terrain_width / 2)
        world_z = (norm_z * self.app.terrain_depth) - (self.app.terrain_depth / 2)

        self.app.plants.rebuild_plants_in_area(world_x, world_z, 8)

    def on_mouse_drag_event(self, x, y, dx, dy):
        # Convert mouse coordinates (window: origin top-left) to texture coordinates (origin bottom-left)
        self.last_mouse = self.app.screen_to_water(x, y)
        if self.target_height is None:
            self._set_target_height()

    def on_mouse_press_event(self, x, y, button):
        # Record the initial mouse position in texture coordinates.
        self.last_mouse = self.app.screen_to_water(x, y)
        self._set_target_height()

    def _set_target_height(self):
        model = self.app.terrain_instance.model

        heights = model.mesh.heights

        x, y = self.last_mouse
        w, h = heights.shape

        ny = round(h * y)
        nx = round(w * x)
        if 0 <= nx < w and 0 <= ny < h:
            self.target_height = heights[ny, nx]
        else:
            self.target_height = None

    def on_mouse_release_event(self, x, y, button):
        self.last_mouse = None


@register_tool
class AnimalPlacementTool:
    """Place and rotate animals using mouse input.

    Left-click to start placement.
    Drag to set the rotation (angle is computed from the drag vector).
    Release to place the animal.
    Scroll to toggle animal type.
    """
    def __init__(self, app: 'WaterApp'):
        from .animals import ANIMAL_Y  # Import to access available animal types
        self.app = app
        self.press_pos: tuple[float, float] | None = None
        self.current_rotation: float = 0.0
        self.animal_types = list(ANIMAL_Y.keys())
        self.current_index = 0
        self.animal_type = self.animal_types[self.current_index]

    def update(self, dt):
        pass

    def on_mouse_press_event(self, x, y, button):
        pos = self.app.screen_to_ground(x, y)
        if pos is not None:
            self.press_pos = pos
            self.animal = self.app.animals.add(self.animal_type, self.press_pos, self.current_rotation)

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.press_pos is None:
            return
        cur_pos = self.app.screen_to_ground(x, y)
        if cur_pos is None:
            return
        d = cur_pos - self.press_pos
        angle = math.atan2(d.x, d.z)
        self.animal.rot = glm.quat(glm.angleAxis(angle, glm.vec3(0, 1, 0)))

    def on_mouse_release_event(self, x, y, button):
        if self.press_pos is None:
            return
        release_pos = self.app.screen_to_ground(x, y)
        if release_pos is None:
            release_pos = self.press_pos

        d = release_pos - self.press_pos
        angle = math.atan2(d.x, d.z)
        self.animal.rot = glm.quat(glm.angleAxis(angle, glm.vec3(0, 1, 0)))
        self.animal = None
        self.press_pos = None

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        # Scroll up toggles forward; scroll down toggles backward.
        if y_offset > 0:
            self.current_index = (self.current_index + 1) % len(self.animal_types)
        elif y_offset < 0:
            self.current_index = (self.current_index - 1) % len(self.animal_types)
        self.animal_type = self.animal_types[self.current_index]
        print(f"Animal selected: {self.animal_type}")
