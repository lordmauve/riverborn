import importlib.resources
from itertools import product
import moderngl
import moderngl_window as mglw
import numpy as np
import noise
from pyrr import Matrix44, Vector3
import imageio as iio

from riverborn import terrain
from riverborn.camera import Camera
from riverborn.shader import load_shader

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


# Helper: create a dummy cube map (each face gets a solid colour).
def create_dummy_cubemap(ctx, size=64):
    # Define six face colors for a simple sky-like environment.
    face_colors = [
        [135, 206, 235],  # Positive X – light sky blue
        [135, 206, 235],  # Negative X
        [135, 206, 250],  # Positive Y – slightly different blue
        [100, 149, 237],  # Negative Y
        [70, 130, 180],  # Positive Z
        [70, 130, 180],  # Negative Z
    ]
    faces = []
    for color in face_colors:
        # Create a face filled with the given color.
        face = np.full((size, size, 3), color, dtype=np.uint8)
        faces.append(face.tobytes())
    # Create a cube map texture.
    cube = ctx.texture_cube((size, size), 3, data=None)
    for i in range(6):
        cube.write(i, faces[i])
    cube.build_mipmaps()
    return cube


class WaterApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Water Plane with GPU Height Map, Cube Map & Depth-based Transparency"
    window_size = (800, 600)
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.terrain = Instance(
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

        # ------------------------------
        # Offscreen framebuffer for water-bottom pass.
        # ------------------------------
        self.offscreen_color = self.ctx.texture(self.wnd.size, components=4)
        self.offscreen_depth = self.ctx.depth_texture(self.wnd.size)
        self.offscreen_fbo = self.ctx.framebuffer(
            color_attachments=[self.offscreen_color],
            depth_attachment=self.offscreen_depth,
        )

        # ------------------------------
        # Create geometries
        # ------------------------------
        # Water-bottom geometry: a large quad (simulate the underwater terrain)
        self.bottom_size = 100.0
        bottom_vertices, bottom_indices = create_quad(self.bottom_size)
        self.bottom_vbo = self.ctx.buffer(bottom_vertices.tobytes())
        self.bottom_ibo = self.ctx.buffer(bottom_indices.tobytes())
        self.bottom_vao = self.ctx.vertex_array(
            self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec3 in_position;
                    in vec2 in_uv;
                    uniform mat4 mvp;
                    varying vec2 v_uv;
                    void main() {
                        v_uv = in_uv;
                        gl_Position = mvp * vec4(in_position, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330
                    varying vec2 v_uv;
                    out vec4 f_color;
                    void main() {
                        // A simple brownish colour for the bottom.
                        f_color = vec4(0.2, 0.15, 0.1, 1.0);
                    }
                """,
            ),
            [(self.bottom_vbo, "3f 2x4", "in_position")],
            index_buffer=self.bottom_ibo,
        )

        # Water plane geometry: a quad covering the same region.
        self.water_size = 100.0
        water_vertices, water_indices = create_quad(self.water_size)
        self.water_vbo = self.ctx.buffer(water_vertices.tobytes())
        self.water_ibo = self.ctx.buffer(water_indices.tobytes())

        # ------------------------------
        # Create water shader program.
        # ------------------------------
        self.water_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec2 in_uv;
                uniform mat4 mvp;
                uniform mat4 model;
                out vec2 v_uv;
                out vec3 v_world;
                void main() {
                    v_uv = in_uv;
                    vec4 world = model * vec4(in_position, 1.0);
                    v_world = world.xyz;
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                // Water plane fragment shader.
                in vec2 v_uv;
                in vec3 v_world;
                uniform sampler2D height_map;   // Height map for ripples.
                uniform samplerCube env_cube;     // Environment cube map.
                uniform sampler2D depth_tex;      // Depth texture from water-bottom pass.
                uniform vec3 camera_pos;
                uniform vec2 resolution;
                uniform float near;
                uniform float far;
                uniform vec3 base_water;
                uniform float water_opaque_depth;
                out vec4 f_color;

                // Function to linearize a non-linear depth value.
                float linearizeDepth(float depth) {
                    return (2.0 * near * far) / (far + near - depth * (far - near));
                }

                void main() {
                    // --- Compute perturbed normal from the height map.
                    float h = texture(height_map, v_uv).r;
                    // Derivatives of the height value (for a simple normal perturbation).
                    float dFdx_h = dFdx(h);
                    float dFdy_h = dFdy(h);
                    float ripple_scale = 1.0;
                    vec3 perturbed_normal = normalize(vec3(-dFdx_h * ripple_scale, 1.0, -dFdy_h * ripple_scale));

                    // --- Fresnel term.
                    vec3 view_dir = normalize(camera_pos - v_world);
                    float fresnel = pow(1.0 - max(dot(perturbed_normal, view_dir), 0.0), 5.0);

                    // --- Reflection from the environment.
                    vec3 refl_dir = reflect(-view_dir, perturbed_normal);
                    vec4 refl_color = vec4(texture(env_cube, refl_dir).rgb, 1.0);

                    // --- Depth-based transparency.
                    // Compute screen-space coordinates.
                    vec2 screen_uv = gl_FragCoord.xy / resolution;
                    // Sample the water-bottom depth (non-linear depth).
                    float scene_depth = texture(depth_tex, screen_uv).r;
                    // Linearize the depths.
                    float scene_lin = linearizeDepth(scene_depth);
                    float water_lin = linearizeDepth(gl_FragCoord.z);
                    float depth_diff = scene_lin - water_lin;
                    // When the water is shallow (small depth difference) we want more transparency.
                    float shallow = clamp(depth_diff / water_opaque_depth, 0.0, 1.0);

                    // Mix: reflection atop base water colour.
                    vec4 diffuse = vec4(base_water, shallow);
                    f_color = mix(diffuse, refl_color, fresnel);
                }
            """,
        )
        # Create a VAO for the water plane.
        self.water_vao = self.ctx.vertex_array(
            self.water_prog,
            [(self.water_vbo, "3f 2f", "in_position", "in_uv")],
            self.water_ibo,
        )

        # ------------------------------
        # Create the water ripple height map texture.
        # ------------------------------
        height_map_data = generate_height_map(256, 256, scale=0.1, octaves=4, base=24)
        self.height_map_tex = self.ctx.texture((256, 256), 3, height_map_data.tobytes())
        self.height_map_tex.build_mipmaps()
        self.height_map_tex.use(location=0)
        self.water_prog["height_map"].value = 0

        # ------------------------------
        # Create a dummy environment cube map.
        # ------------------------------
        self.env_cube = load_env_map(self.ctx)
        self.env_cube.use(location=1)
        self.water_prog["env_cube"].value = 1

        # The depth texture from the offscreen FBO will be bound to unit 2.
        self.offscreen_depth.use(location=2)
        self.water_prog["depth_tex"].value = 2

        # ------------------------------
        # Set up common uniforms.
        # ------------------------------
        self.water_prog["near"].value = 0.1
        self.water_prog["far"].value = 1000.0
        self.water_prog["resolution"].value = self.wnd.size

        # Define model matrices.
        # Water plane: a translation upward to water_level.
        self.water_model = Matrix44.from_translation([0.0, 1.0, 0.0])
        # Water-bottom: assume at y = 0.
        self.bottom_model = Matrix44.from_translation([0.0, 0.0, 0.0])
        self.water_prog["model"].write(self.water_model.astype("f4").tobytes())
        self.bottom_vao.program["mvp"].write(Matrix44.identity(dtype="f4"))

        # Set up a basic camera.
        self.camera_pos = Vector3([0.0, 50.0, 100.0])
        self.look_at = Vector3([0.0, 0.0, 0.0])

        # Enable blending for transparency in the water pass.
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def on_render(self, time, frame_time):
        self.camera.set_aspect(self.wnd.aspect_ratio)

        # ------------------------------
        # First pass: Render water-bottom scene into offscreen framebuffer.
        # ------------------------------
        self.offscreen_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.terrain.render(self.camera)
        # For water bottom, we simply compute an mvp.
        # bottom_model = self.bottom_model
        # mvp_bottom = proj * view * bottom_model
        # self.bottom_vao.program["mvp"].write(mvp_bottom.astype("f4").tobytes())
        # self.bottom_vao.render()

        # ------------------------------
        # Second pass: Render water plane to the default framebuffer.
        # ------------------------------
        self.ctx.screen.use()
        self.ctx.clear(0.2, 0.3, 0.4, 1.0)
        self.terrain.render(self.camera)
        self.height_map_tex.use(location=0)
        self.water_prog["height_map"].value = 0
        self.env_cube.use(location=1)
        self.water_prog["env_cube"].value = 1
        self.offscreen_depth.use(location=2)
        self.water_prog["depth_tex"].value = 2

        self.water_prog['base_water'] = (0.3, 0.25, 0.2)
        self.water_prog['water_opaque_depth'] = 3

        self.camera.bind(self.water_prog, self.water_model, mvp_uniform="mvp", pos="camera_pos")
        self.water_prog["resolution"].value = self.wnd.size
        self.water_vao.render()

    def on_resize(self, width: int, height: int):
        # When the window is resized, update the offscreen framebuffer and resolution uniform.
        self.offscreen_color = self.ctx.texture((width, height), components=4)
        self.offscreen_depth = self.ctx.depth_texture((width, height))
        self.offscreen_fbo = self.ctx.framebuffer(
            color_attachments=[self.offscreen_color],
            depth_attachment=self.offscreen_depth,
        )
        self.water_prog["resolution"].value = (width, height)


def main():
    mglw.run_window_config(WaterApp)
