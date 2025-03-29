import atexit
import heapq
import importlib.resources
from itertools import product
import json
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
import pyglet.media

from riverborn.shadow_debug import render_small_shadow_map

from .tools import TOOLS
from . import picking
from .camera import Camera
from .plants import PlantGrid
from .scene import Light, Material, Scene
from .shader import load_shader
from .ripples import WaterSimulation
from .heightfield import create_noise_texture
from .animals import Animals

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


class SplashScreen:
    def __init__(self, ctx, app):
        self.ctx = ctx
        self.active = True
        self.opacity = 1.0
        self.fading = False
        self.app = app

        # Load the logo texture
        logo_path = importlib.resources.files('riverborn') / "textures/riverborn.png"
        with logo_path.open('rb') as f:
            logo_img = iio.imread(f)

        logo_img = np.flipud(logo_img)

        # Create texture from the image
        self.logo_width, self.logo_height = logo_img.shape[1::-1]
        self.logo_texture = ctx.texture((self.logo_width, self.logo_height), 4)
        self.logo_texture.write(logo_img.tobytes())

        # Aspect ratio of the logo
        self.logo_aspect_ratio = self.logo_width / self.logo_height

        # Create a quad for displaying the logo
        self.logo_vao = geometry.quad_fs(normals=False)

        # Create a shader for rendering the logo with opacity and proper aspect ratio
        self.logo_shader = ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec2 in_texcoord_0;
                out vec2 uv;
                void main() {
                    gl_Position = vec4(in_position, 1.0);
                    uv = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D logo_texture;
                uniform float opacity;
                uniform vec2 screen_size;
                uniform vec2 logo_size;
                in vec2 uv;
                out vec4 fragColor;

                void main() {
                    // Calculate aspect ratios
                    float screen_aspect = screen_size.x / screen_size.y;
                    float logo_aspect = logo_size.x / logo_size.y;

                    // Determine the scaling factors to maintain aspect ratio
                    vec2 scale;
                    if (screen_aspect > logo_aspect) {
                        // Screen is wider than logo
                        scale = vec2(logo_aspect / screen_aspect, 1.0);
                    } else {
                        // Screen is taller than logo
                        scale = vec2(1.0, screen_aspect / logo_aspect);
                    }

                    // Scale logo to 40% of screen height
                    scale *= 0.4;

                    // Calculate new UV coordinates
                    vec2 centered_uv = (uv - 0.5) / scale + 0.5;

                    // Check if the pixel is within the logo bounds
                    if (centered_uv.x < 0.0 || centered_uv.x > 1.0 ||
                        centered_uv.y < 0.0 || centered_uv.y > 1.0) {
                        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
                    } else {
                        vec4 color = texture(logo_texture, centered_uv);
                        fragColor = vec4(color.rgb, color.a * opacity);
                    }
                }
            '''
        )

        # Initial screen size will be updated in render
        self.screen_size = (1.0, 1.0)

    def start_fade_out(self):
        """Start the fade-out animation"""
        if self.active and not self.fading:
            self.fading = True

            async def fade():
                await clock.default_clock.animate(self, 'accel_decel', 1.0, opacity=0.0)
                self.active = False
                self.app.voiceover('start')
                await clock.default_clock.coro.sleep(60)
                if len(self.app.spotted_animals) < 3:
                    self.app.voiceover('2min')
                await clock.default_clock.coro.sleep(60)
                if len(self.app.spotted_animals) < 3:
                    self.app.voiceover('1min')
                await clock.default_clock.coro.sleep(60)
                if len(self.app.spotted_animals) < 3:
                    self.app.voiceover('failed')

            loop.do(fade())

    def render(self):
        """Render the splash screen if active"""
        if not self.active:
            return

        # Update screen size uniform
        self.screen_size = self.ctx.fbo.size

        self.logo_texture.use(0)
        self.logo_shader['logo_texture'] = 0
        self.logo_shader['opacity'] = self.opacity
        self.logo_shader['screen_size'] = self.screen_size
        self.logo_shader['logo_size'] = (self.logo_width, self.logo_height)

        # Clear the depth buffer so splash screen renders on top
        self.ctx.screen.color_mask = False, False, False, False
        self.ctx.clear(depth=1.0)
        self.ctx.screen.color_mask = True, True, True, True

        with self.ctx.scope(enable_only=moderngl.BLEND):
            self.logo_vao.render(self.logo_shader)


class WaterApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Riverborn"
    window_size = (1920, 1080)
    aspect_ratio = None  # Let the window determine the aspect ratio.
    resizable = False

    # FIXME: need to use package data
    resource_dir = Path(__file__).parent

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.tool = TOOLS[self.tool_id](self)

        # Create splash screen
        self.splash_screen = SplashScreen(self.ctx, self)

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
            self.animals.save()

        # Create an instance of the terrain model
        self.terrain_instance = self.scene.add(terrain_model)

        self.plants = PlantGrid(self.scene, terrain_model.mesh)
        self.plants.setup()

        default_material = Material(
            double_sided=False,
            translucent=False,
            transmissivity=0.0,
            receive_shadows=True,
            cast_shadows=True,
            alpha_test=False,
        )

        canoe_model = self.scene.load_wavefront('boat.obj', material=default_material, capacity=1)
        self.canoe = self.scene.add(canoe_model)
        self.canoe.pos = glm.vec3(-100, 0, -100)

        oar_model = self.scene.load_wavefront('oar.obj', material=default_material, capacity=1)
        self.oar = self.scene.add(oar_model)
        self.oar.local_pos = glm.vec3(0, 0, -1)
        self.oar.local_rot = glm.quat()
        files = self.files = importlib.resources.files()
        with (files / 'sounds/splash2.wav').open('rb') as f:
            self.paddle_sound = pyglet.media.load('splash2.wav', f, streaming=False)

        self.music = (files / 'sounds/ambient.ogg').open('rb')
        pyglet.media.load('ambient', self.music, streaming=True).play()

        self.animals = Animals(self.scene)
        self.animals.load()

        # Track which animals have been spotted
        self.spotted_animals = set()

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

    def voiceover(self, name):
        with (self.files / f'sounds/{name}.wav').open('rb') as f:
            pyglet.media.load(f'{name}.wav', f, streaming=False).play()

    canoe_pos = vec2(-80, -80)
    canoe_rot = 0
    canoe_vel = vec2(0, 0)
    canoe_angular_vel = 0
    canoe_pos3 = glm.vec3()
    CANOE_SIZE = 5

    paddle_task = None

    def paddle(self, side: float) -> None:
        """Paddle in the water."""
        async def paddle():
            await clock.default_clock.animate(self.oar, 'accel_decel', 0.3, local_pos=glm.vec3(-0.8 * side, 0, 1))

            speed = glm.vec3(-0.1 * side, -0.4, -1.5)
            self.paddle_sound.play()

            async for dt in clock.coro.frames_dt(seconds=1):
                self.oar.local_pos += dt * speed
                self.canoe_angular_vel += side * 1.0 * dt
                self.canoe_vel += vec2(0, 1.0).rotated(-self.canoe_rot) * dt

        if self.paddle_task is not None:
            self.paddle_task.cancel()
        self.paddle_task = loop.do(paddle())

    def update(self, dt: float) -> None:
        self.tool.update(dt)
        self.update_canoe(dt)
        m = self.canoe.matrix

        back = m * glm.vec3(0, 0, 1)
        front = m * glm.vec3(0, 0, -1)

        self.water_sim.disturb(
            self.pos_to_water(back),
            self.pos_to_water(front),
        )

        self.oar.pos = self.canoe.matrix *  self.oar.local_pos

        self.camera.eye = self.canoe.pos + glm.vec3(0, 15, -20)
        self.camera.look_at(self.canoe.pos)

        # Check for nearby animals that haven't been spotted yet
        self.check_nearby_animals()

    def check_nearby_animals(self) -> None:
        """Check if there are any unspotted animals nearby."""
        # Detection range
        detection_range = 12.0

        # Get canoe position as a 2D point (x, z)
        canoe_pos_2d = (self.canoe.pos.x, self.canoe.pos.z)

        # Check each animal type and its instances
        for animal_type, instances in self.animals.animals.items():
            # Skip if we've already spotted this type
            if animal_type in self.spotted_animals:
                continue

            # Check each instance of this animal type
            for instance in instances:
                # Get animal position as a 2D point (x, z)
                animal_pos_2d = (instance.pos.x, instance.pos.z)

                # Calculate distance (in 2D, ignoring y-axis)
                dx = canoe_pos_2d[0] - animal_pos_2d[0]
                dz = canoe_pos_2d[1] - animal_pos_2d[1]
                distance = math.sqrt(dx*dx + dz*dz)

                # If within range, mark as spotted and call the event handler
                if distance <= detection_range:
                    self.spotted_animals.add(animal_type)
                    loop.do(self.on_animal_spotted(animal_type))
                    break  # No need to check other instances of this type

    async def on_animal_spotted(self, animal_type: str) -> None:
        """Called when the player sees a new animal type for the first time."""
        print(f"Spotted a {animal_type} for the first time!")
        self.voiceover(animal_type)
        if len(self.spotted_animals) == 3:
            await clock.default_clock.coro.sleep(3)
            self.voiceover('win')

        # You can add more visual/audio feedback here, like:
        # - Display a notification
        # - Play a discovery sound
        # - Update a journal or achievement system

    def update_canoe(self, dt):
        self.canoe_vel *= 0.8 ** dt
        current_height = self.terrain_instance.model.mesh.get_terrain_height(
            self.canoe_pos
        )

        # Compute the candidate new position based on current velocity
        candidate_pos = self.canoe_pos + self.canoe_vel * dt

        # Get terrain height at the candidate position.
        # Note: self.canoe_pos is a vec2 where x is world x and y is world z.
        terrain_height = self.terrain_instance.model.mesh.get_terrain_height(
            candidate_pos
        )

        # Check if the terrain is too high (land, where height > 1)
        if terrain_height > 1 and terrain_height >= current_height:
            # Collision: cancel or reduce the movement.
            # Here, we simply dampen the velocity further.
            self.canoe_vel *= 1 / (terrain_height + 1) ** dt
            # Optionally, you might choose not to update self.canoe_pos at all.
            self.canoe_pos += self.canoe_vel * dt
            terrain_height = self.terrain_instance.model.mesh.get_terrain_height(
                self.canoe_pos
            )
        else:
            # No collision: update the canoe's position
            self.canoe_pos = candidate_pos

        # Update angular velocity and rotation as usual
        self.canoe_angular_vel *= 0.3 ** dt
        self.canoe_rot += self.canoe_angular_vel * dt

        # Update the display model. We keep a constant water level of y=1.
        self.canoe.pos = glm.vec3(self.canoe_pos.x, max(terrain_height, 1), self.canoe_pos.y)
        self.canoe.rot = glm.quat(glm.angleAxis(self.canoe_rot, glm.vec3(0, 1, 0)))

    def on_render(self, time, frame_time):
        tick_loop(frame_time)
        self.update(frame_time)
        self.camera.set_aspect(self.wnd.aspect_ratio)

        self.water_sim.simulate()
        # ------------------------------
        # First pass: Render scene into offscreen framebuffer.
        # ------------------------------
        with self.ctx.scope(framebuffer=self.offscreen_fbo, enable=moderngl.DEPTH_TEST):
            self.ctx.clear(0.6, 0.7, 1.0, 1.0)
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

        # Render splash screen on top if active
        if self.splash_screen.active:
            self.splash_screen.render()

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

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        if handler := getattr(self.tool, 'on_mouse_scroll_event', None):
            handler(x_offset, y_offset)

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
        # If splash screen is active and space is pressed, start fade-out
        keys = self.wnd.keys
        if action == self.wnd.keys.ACTION_PRESS and key == keys.SPACE and self.splash_screen.active:
            self.splash_screen.start_fade_out()
            return

        op = 'press' if action == self.wnd.keys.ACTION_PRESS else 'release'
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
