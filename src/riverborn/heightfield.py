"""Heightfield module for generating and displaying procedural terrain."""
import moderngl
import moderngl_window as mglw
import numpy as np
import noise
from pyglm import glm

from .terrain import make_terrain
from .shader import load_shader
from .camera import Camera
from .scene import Light, Scene, TerrainModel


def create_noise_texture(size: int = 256, color=(1.0, 1.0, 1.0)):
    """Generate a texture with Perlin noise."""
    tex_width, tex_height = size, size
    texture_data = np.zeros((tex_height, tex_width, 3), dtype=np.uint8)
    texture_noise_scale = 0.05
    for i in range(tex_height):
        for j in range(tex_width):
            t = noise.pnoise2(
                j * texture_noise_scale,
                i * texture_noise_scale,
                octaves=4,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=1024,
                repeaty=1024,
                base=24,
            )
            c = (t + 1) * 0.5
            texture_data[i, j] = tuple(int(c * comp * 255) for comp in color)
    # Flip vertically to account for texture coordinate differences.
    texture_data = np.flipud(texture_data)
    return texture_data


class HeightfieldApp(mglw.WindowConfig):
    """Demo application for displaying a heightfield using the Scene framework."""
    gl_version = (3, 3)
    title = "Textured Heightfield with ModernGL"
    window_size = (800, 600)
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize scene
        self.scene = Scene()

        self.light = Light(
            direction=[0.5, -0.8, -0.3],
            color=[1.0, 1.0, 1.0],
            ambient=[0.2, 0.2, 0.2],
            ortho_size=60.0,
            near=1.0,
            far=200.0,
            target=glm.vec3(0.0, 0.0, 0.0)
        )

        # Create a shader program for the terrain
        terrain_shader = load_shader('diffuse', defines={'INSTANCED': '1'})

        # Generate terrain texture
        terrain_texture = create_noise_texture(size=512, color=(0.6, 0.5, 0.4))

        # Create a terrain model and add it to the scene
        terrain_model = self.scene.create_terrain(
            'terrain',
            terrain_shader,
            segments=100,
            width=100,
            depth=100,
            height=10,
            noise_scale=0.1,
            texture=terrain_texture
        )

        # Create an instance of the terrain model
        self.terrain_instance = self.scene.add(terrain_model)

        # Create the camera
        self.camera = Camera(
            eye=[0.0, 50.0, 100.0],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
            fov=70.0,
            aspect=self.wnd.aspect_ratio,
            near=0.1,
            far=1000.0,
        )

        self.time = 0.0

    def on_render(self, time, frame_time):
        self.ctx.clear(0.2, 0.4, 0.6)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.time += frame_time

        # Update terrain rotation
        self.terrain_instance.rot = glm.quat(
            glm.vec3(0.0, self.time, 0.0)
        )

        # Update camera aspect if needed
        self.camera.set_aspect(self.wnd.aspect_ratio)

        # Draw the scene
        self.scene.draw(self.camera, self.light)

    def on_close(self):
        self.scene.destroy()


def main():
    mglw.run_window_config(HeightfieldApp)
