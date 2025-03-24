import moderngl
import moderngl_window as mglw
import numpy as np
import noise
from pyrr import Matrix44, Quaternion, Vector3

from . import terrain
from .shader import load_shader
from .camera import Camera


def create_noise_texture(size: int = 256, color=(1.0, 1.0, 1.0)):
    """Generate a 256x256 texture with Perlin noise."""
    ctx = mglw.ctx()
    tex_width, tex_height = size, size
    texture_data = np.zeros((tex_height, tex_width, 3), dtype=np.uint8)
    texture_noise_scale = 0.1
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
    texture = ctx.texture(
        (tex_width, tex_height), 3, texture_data.tobytes()
    )
    texture.build_mipmaps()
    texture.use(location=0)
    return texture


class HeightfieldApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Textured Heightfield with ModernGL"
    window_size = (800, 600)
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.instance = Instance(
            self.ctx,
            terrain.make_terrain(100, 100, 100, 10, 0.1),
            load_shader(self.ctx, 'diffuse'),
            create_noise_texture(self.ctx, color=(0.6, 0.5, 0.4))
        )

        # Create the camera.
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
        self.instance.rotation = Quaternion.from_eulers(
            [0.0, self.time, 0.0], dtype=np.float32
        )

        # Update camera aspect if needed.
        self.camera.set_aspect(self.wnd.aspect_ratio)
        self.instance.render(self.camera)


class Instance:
    vao: moderngl.VertexArray
    prog: moderngl.Program
    texture: moderngl.Texture

    def __init__(self,
        mesh: terrain.Mesh,
        shader: moderngl.Program,
        texture: moderngl.Texture
    ) -> None:
        self.pos = Vector3([0.0, 0.0, 0.0])
        self.rotation = Quaternion()
        self.ctx = mglw.ctx()
        self.mesh = mesh
        self.prog = shader
               # Create buffers: vertex buffer and index buffer.
        self.vbo = self.ctx.buffer(mesh.vertices)
        self.ibo = self.ctx.buffer(mesh.indices.astype("i4").tobytes())

        # Build the shader program.
        self.prog = load_shader('diffuse')

        # Create the vertex array object linking attributes.
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            self.ibo,
        )

        # Create the texture (generated via Perlin noise).
        self.texture = texture

    def render(self, camera: Camera):
        self.texture.use(location=0)
        self.prog["Texture"] = 0
        self.prog["light_dir"].value = (0.5, 1.0, 0.3)
        self.prog["light_color"].value = (1.0, 1.0, 1.0)
        self.prog["ambient_color"].value = (0.3, 0.3, 0.3)

        model = Matrix44.from_translation(self.pos) * self.rotation

        # Bind the camera matrices (MVP and normal matrix) to the shader.
        camera.bind(self.prog, model, normal_matrix="normal_matrix")

        self.vao.render()


def main():
    mglw.run_window_config(HeightfieldApp)
