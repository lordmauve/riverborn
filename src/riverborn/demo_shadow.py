"""
Shadow Mapping Demo - Shows integration with Scene framework with debug visualization
"""
import os
import sys
import random
import math
import moderngl
import moderngl_window as mglw
from pyglm import glm

from riverborn.camera import Camera
from riverborn.scene import Scene, Material
from riverborn.shadow import ShadowSystem, Light
from riverborn.heightfield import create_noise_texture
from riverborn.shadow_debug import render_small_shadow_map


class ShadowMappingDemo(mglw.WindowConfig):
    """Demo that shows shadow mapping with the Scene framework."""
    gl_version = (3, 3)
    title = "Shadow Mapping Demo"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Create the scene
        self.scene = Scene()

        # Create the camera.
        self.camera = Camera(
            eye=[0.0, 20.0, 50.0],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
            fov=70.0,
            aspect=self.wnd.aspect_ratio,
            near=0.1,
            far=1000.0,
        )

        # Create a directional light
        self.light = Light(
            direction=[0.5, -0.8, -0.3],
            color=[1.0, 0.9, 0.8],
            ambient=[0.0, 0.1, 0.0],
            ortho_size=50.0
        )

        # Create the shadow system
        self.shadow_system = ShadowSystem(shadow_map_size=1024)
        self.shadow_system.set_light(self.light)

        # Generate terrain texture
        terrain_texture = create_noise_texture(size=512, color=(0.6, 0.5, 0.4))

        # Create a terrain model and add it to the scene
        # Define terrain material properties using the new Material dataclass
        terrain_material = Material(
            receive_shadows=True
        )

        terrain_model = self.scene.create_terrain(
            'terrain',
            segments=100,
            width=40,
            depth=40,
            height=5,
            noise_scale=0.1,
            texture=terrain_texture,
            material=terrain_material
        )

        # Create an instance of the terrain model
        self.terrain_instance = self.scene.add(terrain_model)

        # Define plant material properties using the new Material dataclass
        # Plants are double-sided and have some translucency
        plant_material = Material(
            double_sided=True,
            translucent=True,
            transmissivity=0.3,
            receive_shadows=True,
            alpha_test=True
        )

        # Load a fern model with appropriate material properties
        self.plant_model = self.scene.load_wavefront('fern.obj', material=plant_material, capacity=50)

        # Create plant instances
        for _ in range(20):
            inst = self.scene.add(self.plant_model)
            # Random position on the terrain
            inst.pos = (random.uniform(-20, 20), 0, random.uniform(-20, 20))
            inst.rotate(random.uniform(0, 2 * math.pi), glm.vec3(0, 1, 0))
            inst.scale = glm.vec3(random.uniform(0.05, 0.1))
            inst.update()

        # Debug mode flag
        self.debug_mode = False

        # Time tracking
        self.time = 0
        self.rotate_light = True

    def on_resize(self, width, height):
        self.camera.set_aspect(width / height)

    def on_render(self, time, frame_time):
        self.time += frame_time
        self.ctx.clear(0.2, 0.3, 0.4)

        # Update light direction if rotating
        if self.rotate_light:
            angle = self.time * 0.5
            self.light.direction = glm.normalize(glm.vec3(
                math.sin(angle),
                -1.2,
                math.cos(angle)
            ))
            self.light.update_matrices()

        # First pass: render scene to shadow map from light's perspective
        self.shadow_system.render_depth(self.scene)

        # Second pass: render scene with shadows
        ctx = mglw.ctx()
        ctx.screen.use()

        # Render scene objects
        self.scene.draw_with_shadows(self.camera, self.light, self.shadow_system)

        render_small_shadow_map(
            *self.wnd.buffer_size,
            self.shadow_system,
            self.light
        )

    def on_key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.ESCAPE:
                sys.exit()
            elif key == self.wnd.keys.SPACE:
                self.rotate_light = not self.rotate_light
                print(f"Light rotation: {'on' if self.rotate_light else 'off'}")
            elif key == self.wnd.keys.D:
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'on' if self.debug_mode else 'off'}")
            elif key == self.wnd.keys.F12:
                from riverborn.screenshot import screenshot
                screenshot()

    def on_mouse_drag_event(self, x, y, dx, dy):
        # Simple camera orbit on mouse drag
        if self.wnd.mouse_states.left:
            sensitivity = 0.005
            self.camera.eye = glm.vec3(
                glm.rotate(
                    glm.mat4(1.0),
                    -dx * sensitivity,
                    glm.vec3(0, 1, 0)
                ) * glm.vec4(self.camera.eye, 1.0)
            )
            self.camera.look_at(glm.vec3(0, 0, 0))


def main():
    """Run the shadow mapping demo."""
    mglw.run_window_config(ShadowMappingDemo)


if __name__ == '__main__':
    main()
