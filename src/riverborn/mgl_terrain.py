import importlib.resources
from itertools import product
import moderngl
import moderngl_window as mglw
import numpy as np
import noise
from pyrr import Matrix44, Vector3
import imageio as iio

from riverborn import picking, terrain
from riverborn.blending import blend_func
from riverborn.camera import Camera
from riverborn.shader import load_shader

from .ripples import WaterSimulation
from .heightfield import Instance, create_noise_texture

# Helper: create a simple quad geometry with positions (3f) and UV coordinates (2f)
def create_quad(size):
    # A quad centered at origin, lying in the XZ plane.
    vertices = np.array(
        [
            # x,     y,    z,    u,   v
            (-size, 0.0, -size, 0.0, 0.0),
            (size, 0.0, -size, 1.0, 0.0),
            (size, 0.0, size, 1.0, 1.0),
            (-size, 0.0, size, 0.0, 1.0),
        ],
        dtype="f4",
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="i4")
    return vertices, indices


# Helper: generate a grayscale height map (256x256) using Perlin noise.
def generate_height_map(width=256, height=256, scale=0.1, octaves=4, base=24):
    data = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            n = noise.pnoise2(
                j * scale,
                i * scale,
                octaves=octaves,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=1024,
                repeaty=1024,
                base=base,
            )
            color = int((n + 1) * 0.5 * 255)
            data[i, j] = (color, color, color)
    # Flip vertically so UVs match OpenGL conventions.
    data = np.flipud(data)
    return data


def load_env_map(ctx: moderngl.Context) -> moderngl.TextureCube:
    """Load a cube map texture from the textures/skybox directory."""
    images = []
    skybox = importlib.resources.files('riverborn') / "textures/skybox"
    for axis, face in product(("x", "y", "z"), ("pos", "neg")):
        file = skybox / f"{face}{axis}.jpg"
        img = iio.imread(file.open("rb"))
        width, height, _ = img.shape
        images.append(img)
    im = np.array(images)
    tex_size = (width, height)
    cube = ctx.texture_cube(tex_size, 3, data=None)
    for i in range(6):
        cube.write(i, images[i])
    cube.build_mipmaps()
    return cube


class WaterApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Water Plane with GPU Height Map, Cube Map & Depth-based Transparency"
    window_size = (800, 600)
    aspect_ratio = None  # Let the window determine the aspect ratio.
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.terrain = Instance(
            terrain.make_terrain(100, 100, 100, 10, 0.1),
            load_shader('diffuse'),
            create_noise_texture(1024, color=(0.6, 0.5, 0.4))
        )

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

        self.offscreen_depth = self.ctx.depth_texture(self.wnd.size)
        self.offscreen_fbo = self.ctx.framebuffer(
            depth_attachment=self.offscreen_depth,
        )

        # Water plane geometry: a quad covering the same region.
        self.water_size = 100.0
        water_vertices, water_indices = create_quad(self.water_size)
        self.water_vbo = self.ctx.buffer(water_vertices.tobytes())
        self.water_ibo = self.ctx.buffer(water_indices.tobytes())

        # ------------------------------
        # Create water shader program.
        # ------------------------------
        self.water_prog = load_shader("water")
        # uniforms = {
        #     name: self.water_prog[name] for name in self.water_prog
        #     if isinstance(self.water_prog[name], moderngl.Uniform)
        # }
        # print(uniforms)
        # Create a VAO for the water plane.
        self.water_vao = self.ctx.vertex_array(
            self.water_prog,
            [(self.water_vbo, "3f 2f", "in_position", "in_uv")],
            self.water_ibo,
        )

        # ------------------------------
        # Create the water ripple height map texture.
        # ------------------------------
        self.water_sim = WaterSimulation(
            1024,
            1024,
        )



        self.env_cube = load_env_map(self.ctx)

        # Define model matrices.
        # Water plane: a translation upward to water_level.
        self.water_model = Matrix44.from_translation([0.0, 1.0, 0.0], dtype='f4')
        # Water-bottom: assume at y = 0.


        # Set up a basic camera.
        self.camera_pos = Vector3([0.0, 50.0, 100.0])
        self.look_at = Vector3([0.0, 0.0, 0.0])


    def on_render(self, time, frame_time):
        self.camera.set_aspect(self.wnd.aspect_ratio)

        self.water_sim.simulate()
        # ------------------------------
        # First pass: Render water-bottom scene into offscreen framebuffer.
        # ------------------------------
        self.offscreen_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.terrain.render(self.camera)

        # ------------------------------
        # Second pass: Render water plane to the default framebuffer.
        # ------------------------------

        self.ctx.screen.use()
        self.ctx.clear(0.2, 0.3, 0.4, 1.0)
        self.terrain.render(self.camera)

        self.water_prog["env_cube"].value = 1
        self.water_prog["depth_tex"].value = 2
        self.water_prog["near"].value = 0.1
        self.water_prog["far"].value = 1000.0
        self.water_prog["resolution"].value = self.wnd.size
        self.water_prog["model"].write(self.water_model)
        self.env_cube.use(location=1)
        self.water_prog["env_cube"].value = 1
        self.offscreen_depth.use(location=2)
        self.water_prog["depth_tex"].value = 2

        self.water_sim.texture.use(location=0)
        self.water_prog["height_map"].value = 0
        self.water_prog['base_water'] = (0.3, 0.25, 0.2)
        self.water_prog['water_opaque_depth'] = 3

        self.camera.bind(self.water_prog, self.water_model, mvp_uniform="mvp", pos="camera_pos")
        x, y, w, h = self.wnd.viewport
        self.water_prog["resolution"].value = self.wnd.size
        self.water_vao.render()

    def on_resize(self, width: int, height: int):
        # When the window is resized, update the offscreen framebuffer and resolution uniform.
        self.offscreen_depth = self.ctx.depth_texture((width, height))
        self.offscreen_fbo = self.ctx.framebuffer(
            depth_attachment=self.offscreen_depth,
        )
        self.water_prog["resolution"].value = (width, height)
                # Create the camera.
        self.camera.set_aspect(width / height)
        self.ctx.gc()

    last_mouse: tuple[float, float] | None = None

    def on_mouse_drag_event(self, x, y, dx, dy):
        # Convert mouse coordinates (window: origin top-left) to texture coordinates (origin bottom-left)
        width, height = self.wnd.size

        ray = picking.get_mouse_ray(self.camera, x, y, width, height)
        intersection = picking.intersect_ray_plane(ray, 0.0)
        if intersection is None:
            self.last_mouse = None
            return

        cur_pos = (intersection[0] / self.water_size, intersection[2] / self.water_size)
        cur_pos = (cur_pos[0] * 0.5 + 0.5, cur_pos[1] * 0.5 + 0.5)

        if self.last_mouse is None:
            self.last_mouse = cur_pos

        # Apply a disturbance by drawing between the last and current mouse positions.
        self.water_sim.disturb(self.last_mouse, cur_pos)
        self.last_mouse = cur_pos

    def on_mouse_press_event(self, x, y, button):
        # Record the initial mouse position in texture coordinates.
        width, height = self.wnd.size
        self.last_mouse = (x / width, 1 - y / height)

    def on_mouse_release_event(self, x, y, button):
        self.last_mouse = None


def main():
    mglw.run_window_config(WaterApp)
