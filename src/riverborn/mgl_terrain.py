import atexit
import heapq
import importlib.resources
from itertools import product
import math
from pathlib import Path
import random
import sys
import moderngl
from moderngl_window import geometry
import moderngl_window as mglw
import numpy as np
import noise
import imageio as iio
from wasabigeom import vec2
from wasabi2d import loop, clock
from pyglm import glm

from riverborn.shadow_debug import render_small_shadow_map

from .tools import TOOLS
from . import picking
from .camera import Camera
from .plants import PlantGrid
from .scene import Light, Material, Scene
from .shader import load_shader
from .ripples import WaterSimulation
from .heightfield import create_noise_texture


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


SUN_DIR = glm.normalize(glm.vec3(1, 1, 1))

# Clear a default PygameEvents.run task
loop._clear_all()


def tick_loop(dt: float) -> None:
    clock.default_clock.tick(dt)

    while loop.runnable:
        task = heapq.heappop(loop.runnable)
        try:
            task._step()
        except SystemExit as e:
            if main:
                exit = e
                main.cancel()
            else:
                raise
    for task in loop.runnable_next:
        task._resume_value = dt
    loop.runnable, loop.runnable_next = loop.runnable_next, loop.runnable


class WaterApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Riverborn"
    window_size = (1920, 1080)
    aspect_ratio = None  # Let the window determine the aspect ratio.
    resizable = False

    # FIXME: need to use package data
    resource_dir = Path(__file__).parent.parent.parent / 'assets_source/'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.tool = TOOLS[self.tool_id](self)

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

        # Create a directional light with shadows enabled
        self.light = Light(
            direction=[0.5, -0.3, 0.3],
            color=[1.0, 0.9, 0.8],
            ambient=[0.0, 0.1, 0.0],
            ortho_size=50.0,
            shadows=True  # Enable shadows (default is True)
        )

        # Generate terrain texture
        terrain_texture = create_noise_texture(size=512, color=(0.6, 0.5, 0.4))

        # Create a terrain model and add it to the scene
        # Define terrain material properties
        terrain_material = Material(
            receive_shadows=True,  # This terrain receives shadows
            cast_shadows=True      # This terrain casts shadows
        )

        self.terrain_width = self.terrain_depth = 200

        try:
            terrain_model = self.scene.load_terrain(
                'data/terrain.npy',
                width=self.terrain_width,
                depth=200,
                texture=terrain_texture,
                material=terrain_material
            )
        except FileNotFoundError:
            terrain_model = self.scene.create_terrain(
                'terrain',
                segments=100,
                width=self.terrain_width,
                depth=self.terrain_depth,
                height=10,
                noise_scale=0.05,
                texture=terrain_texture,
                material=terrain_material
            )
        @atexit.register
        def save_terrain():
            terrain_path = Path(__file__).parent / 'data/terrain.npy'
            terrain_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the terrain model to a file
            with terrain_path.open('wb') as f:
                np.save(f, terrain_model.mesh.heights, allow_pickle=False)

        # Create an instance of the terrain model
        self.terrain_instance = self.scene.add(terrain_model)

        self.plants = PlantGrid(self.scene, terrain_model.mesh)
        self.plants.setup()

        canoe_material = Material(
            double_sided=False,
            translucent=False,
            transmissivity=0.0,
            receive_shadows=True,
            cast_shadows=True,
            alpha_test=False,
        )

        canoe_model = self.scene.load_wavefront('boat.obj', material=canoe_material, capacity=1)
        self.canoe = self.scene.add(canoe_model)
        self.canoe.pos = glm.vec3(0, 0, 0)

        oar_model = self.scene.load_wavefront('oar.obj', material=canoe_material, capacity=1)
        self.oar = self.scene.add(oar_model)
        self.oar.local_pos = glm.vec3(0, 0, -1)
        self.oar.local_rot = glm.quat()

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
        self.water_model = glm.translate(glm.vec3([0.0, 1.0, 0.0]))
        # Water-bottom: assume at y = 0.

        self.copy_vao = geometry.quad_fs(normals=False)

        self.on_resize(*self.wnd.size)

    canoe_pos = vec2(0, 0)
    canoe_rot = 0
    canoe_vel = vec2(0, 0)
    canoe_angular_vel = 0
    canoe_pos3 = glm.vec3()
    CANOE_SIZE = 5

    paddle_task = None

    def paddle(self, side: float) -> None:
        """Paddle in the water."""
        async def paddle():
            await clock.default_clock.animate(self.oar, 'accel_decel', 0.3, local_pos=glm.vec3(-side, 0, 1))

            speed = glm.vec3(0, -0.4, -2)

            async for dt in clock.coro.frames_dt(seconds=1):
                self.oar.local_pos += dt * speed
                self.canoe_angular_vel += side * 1.0 * dt
                self.canoe_vel += vec2(0, 1.0).rotated(-self.canoe_rot) * dt

        if self.paddle_task is not None:
            self.paddle_task.cancel()
        self.paddle_task = loop.do(paddle())

    def update(self, dt: float) -> None:
        self.tool.update(dt)
        self.canoe_vel *= 0.6 ** dt
        self.canoe_pos += self.canoe_vel * dt
        self.canoe_angular_vel *= 0.3 ** dt
        self.canoe_rot += self.canoe_angular_vel * dt

        self.canoe.pos = glm.vec3(self.canoe_pos.x, 1, self.canoe_pos.y)
        self.canoe.rot = glm.quat(
            glm.angleAxis(self.canoe_rot, glm.vec3(0, 1, 0))
        )
        m = self.canoe.matrix

        back = m * glm.vec3(0, 0, 0.4)
        front = m * glm.vec3(0, 0, -0.4)

        self.water_sim.disturb(
            self.pos_to_water(back),
            self.pos_to_water(front),
        )

        self.oar.pos = self.canoe.matrix *  self.oar.local_pos

        self.camera.eye = self.canoe.pos + glm.vec3(0, 15, -20)
        self.camera.look_at(self.canoe.pos)

    def on_render(self, time, frame_time):
        tick_loop(frame_time)
        self.update(frame_time)
        self.camera.set_aspect(self.wnd.aspect_ratio)

        self.water_sim.simulate()
        # ------------------------------
        # First pass: Render scene into offscreen framebuffer.
        # ------------------------------
        with self.ctx.scope(framebuffer=self.offscreen_fbo, enable=moderngl.DEPTH_TEST):
            self.ctx.clear(0.2, 0.2, 0.4, 1.0)
            self.scene.draw(self.camera, self.light)

        copy_shader = load_shader("copy")
        copy_shader.bind(
            input_texture=self.offscreen_color,
        )
        with self.ctx.scope(enable_only=moderngl.NOTHING):
            self.ctx.depth_mask = False
            self.copy_vao.render(copy_shader)
            self.ctx.depth_mask = True

        self.water_prog["env_cube"].value = 1
        self.water_prog["depth_tex"].value = 2
        self.water_prog["near"].value = 0.1
        self.water_prog["far"].value = 1000.0
        self.water_prog["resolution"].value = self.wnd.size
        self.water_prog["m_model"].write(self.water_model)
        self.env_cube.use(location=1)
        self.water_prog["env_cube"].value = 1
        self.offscreen_depth.use(location=2)
        self.water_prog["depth_tex"].value = 2

        self.water_sim.texture.use(location=0)
        self.water_prog["height_map"].value = 0
        self.water_prog['base_water'] = (0.2, 0.15, 0.1)
        self.water_prog['water_opaque_depth'] = 3


        self.camera.bind(self.water_prog, pos_uniform="camera_pos")
        x, y, w, h = self.wnd.viewport
        self.water_prog["resolution"].value = self.wnd.size
        with self.ctx.scope(enable_only=moderngl.BLEND):
            self.water_vao.render()
        if self.recorder is not None:
            self.recorder._vid_frame()

        # Display a small shadow map preview
        # if self.light.shadows and self.light.shadow_system:
        #     render_small_shadow_map(
        #         *self.wnd.buffer_size,
        #         self.offscreen_depth,
        #         self.camera
        #     )

    def on_resize(self, width: int, height: int):
        # When the window is resized, update the offscreen framebuffer and resolution uniform.
        self.offscreen_depth = self.ctx.depth_texture((width, height))
        self.offscreen_color = self.ctx.texture((width, height), 4)
        self.offscreen_fbo = self.ctx.framebuffer(
            color_attachments=[self.offscreen_color],
            depth_attachment=self.offscreen_depth,
        )
        self.water_prog["resolution"].value = (width, height)
        # Create the camera.
        self.camera.set_aspect(width / height)
        self.ctx.gc()

    def screen_to_ground(self, x, y) -> glm.vec3 | None:
        width, height = self.wnd.size
        ray = picking.get_mouse_ray(self.camera, x, y, width, height)
        intersection = picking.intersect_ray_plane(ray, 0.0)
        if intersection is None:
            return None

        return intersection

    def pos_to_water(self, pos: glm.vec3) -> tuple[float, float]:
        cur_pos = (pos[0] / self.water_size, pos[2] / self.water_size)
        cur_pos = (cur_pos[0] * 0.5 + 0.5, cur_pos[1] * 0.5 + 0.5)
        return cur_pos

    def screen_to_water(self, x: float, y: float) -> tuple[float, float] | None:
        intersection = self.screen_to_ground(x, y)
        return intersection and self.pos_to_water(intersection)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.tool.on_mouse_drag_event(x, y, dx, dy)

    def on_mouse_press_event(self, x, y, button):
        self.tool.on_mouse_press_event(x, y, button)

    def on_mouse_release_event(self, x, y, button):
        self.tool.on_mouse_release_event(x, y, button)

    recorder = None

    tool_id = 0

    def mount_tool(self):
        """Mount the tool to the application."""
        cls = TOOLS[self.tool_id]
        print(cls.__doc__ or cls.__name__)
        self.tool = cls(self)

    def on_key_event(self, key, action, modifiers):

        op = 'press' if action == self.wnd.keys.ACTION_PRESS else 'release'
        keys = self.wnd.keys
        match op, key, modifiers.shift:
            case ('press', keys.ESCAPE, _):
                sys.exit()

            case ('press', keys.TAB, shift):
                self.tool_id = (self.tool_id + (-1 if shift else 1)) % len(TOOLS)
                self.mount_tool()

            case ('press', keys.F12, False):
                from .screenshot import screenshot
                screenshot()

            case ('press', keys.F12, True):
                if self.recorder is None:
                    from .screenshot import VideoRecorder
                    self.recorder = VideoRecorder()
                self.recorder.toggle_recording()

            case 'press', keys.LEFT, _:
                self.paddle(-1)

            case 'press', keys.RIGHT, _:
                self.paddle(1)




def main():
    mglw.run_window_config(WaterApp)
