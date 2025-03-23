import moderngl
import moderngl_window as mglw
import numpy as np
import noise
from pyrr import Matrix44, Vector3


class HeightfieldApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Textured Heightfield with ModernGL"
    window_size = (800, 600)
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Grid and noise parameters
        self.grid_width = 100.0
        self.grid_depth = 100.0
        self.segments = 100
        self.noise_scale = 0.1
        self.height_multiplier = 10.0

        # Build the grid geometry (positions, normals, uv's)
        self.create_grid()

        # Create buffers: vertex buffer and index buffer.
        self.vbo = self.ctx.buffer(self.vertices)
        self.ibo = self.ctx.buffer(self.indices.astype("i4").tobytes())

        # Build the shader program.
        self.prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec3 in_position;
            in vec3 in_normal;
            in vec2 in_uv;

            uniform mat4 mvp;
            uniform mat3 normal_matrix;

            out vec3 frag_normal;
            out vec2 frag_uv;

            void main() {
                frag_uv = in_uv;
                frag_normal = normalize(normal_matrix * in_normal);
                gl_Position = mvp * vec4(in_position, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            in vec3 frag_normal;
            in vec2 frag_uv;

            uniform sampler2D Texture;
            uniform vec3 light_dir;
            uniform vec3 light_color;
            uniform vec3 ambient_color;

            out vec4 fragColor;

            void main() {
                vec3 normal = normalize(frag_normal);
                vec3 light = normalize(light_dir);
                float diffuse = max(dot(normal, light), 0.0);
                vec3 tex_color = texture(Texture, frag_uv).rgb;
                vec3 color = ambient_color * tex_color + diffuse * light_color * tex_color;
                fragColor = vec4(color, 1.0);
            }
            """,
        )

        # Create the vertex array object linking attributes.
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
            self.ibo,
        )

        # Create the texture (generated via Perlin noise).
        self.create_texture()

        # Set initial lighting uniforms.
        self.prog["light_dir"].value = (0.5, 1.0, 0.3)
        self.prog["light_color"].value = (1.0, 1.0, 1.0)
        self.prog["ambient_color"].value = (0.3, 0.3, 0.3)

        self.model = Matrix44.identity()
        self.time = 0.0

    def create_grid(self):
        # Create a grid of (segments+1) x (segments+1) vertices.
        num_vertices = (self.segments + 1) * (self.segments + 1)
        self.vertices = np.zeros(
            num_vertices,
            dtype=[
                ("in_position", np.float32, 3),
                ("in_normal", np.float32, 3),
                ("in_uv", np.float32, 2),
            ],
        )

        xs = np.linspace(-self.grid_width / 2, self.grid_width / 2, self.segments + 1)
        zs = np.linspace(-self.grid_depth / 2, self.grid_depth / 2, self.segments + 1)
        dx = xs[1] - xs[0]
        dz = zs[1] - zs[0]

        # First, compute heights using Perlin noise.
        heights = np.zeros((self.segments + 1, self.segments + 1), dtype=np.float32)
        for i, z in enumerate(zs):
            for j, x in enumerate(xs):
                n = noise.pnoise2(
                    x * self.noise_scale,
                    z * self.noise_scale,
                    octaves=4,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=1024,
                    repeaty=1024,
                    base=42,
                )
                h = n * self.height_multiplier
                heights[i, j] = h
                idx = i * (self.segments + 1) + j
                self.vertices["in_position"][idx] = (x, h, z)
                self.vertices["in_uv"][idx] = (j / self.segments, i / self.segments)

        # Compute normals using finite (central) differences.
        vertex_normals = np.zeros(
            (self.segments + 1, self.segments + 1, 3), dtype=np.float32
        )
        for i in range(self.segments + 1):
            for j in range(self.segments + 1):
                hL = heights[i, j - 1] if j > 0 else heights[i, j]
                hR = heights[i, j + 1] if j < self.segments else heights[i, j]
                hD = heights[i - 1, j] if i > 0 else heights[i, j]
                hU = heights[i + 1, j] if i < self.segments else heights[i, j]
                dX = (hR - hL) / (2 * dx)
                dZ = (hU - hD) / (2 * dz)
                n = np.array([-dX, 1.0, -dZ], dtype=np.float32)
                n /= np.linalg.norm(n)
                vertex_normals[i, j] = n

        # Assign normals to our vertex array.
        for i in range(self.segments + 1):
            for j in range(self.segments + 1):
                idx = i * (self.segments + 1) + j
                self.vertices["in_normal"][idx] = vertex_normals[i, j]

        # Build indices for drawing triangles.
        indices = []
        for i in range(self.segments):
            for j in range(self.segments):
                top_left = i * (self.segments + 1) + j
                top_right = top_left + 1
                bottom_left = (i + 1) * (self.segments + 1) + j
                bottom_right = bottom_left + 1
                indices.extend(
                    [
                        top_left,
                        bottom_left,
                        top_right,
                        top_right,
                        bottom_left,
                        bottom_right,
                    ]
                )
        self.indices = np.array(indices, dtype=np.int32)

    def create_texture(self):
        # Generate a 256x256 texture with Perlin noise.
        tex_width, tex_height = 256, 256
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
                color = int((t + 1) * 0.5 * 255)
                texture_data[i, j] = (color, color, color)
        # Flip vertically to account for texture coordinate differences.
        texture_data = np.flipud(texture_data)
        self.texture = self.ctx.texture(
            (tex_width, tex_height), 3, texture_data.tobytes()
        )
        self.texture.build_mipmaps()
        self.texture.use(location=0)
        self.prog["Texture"] = 0

    def on_render(self, time, frame_time):
        self.ctx.clear(0.2, 0.4, 0.6)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.time += frame_time
        # Rotate the model over time.
        self.model = Matrix44.from_y_rotation(self.time * 0.5)

        # Set up a basic camera.
        eye = Vector3([0.0, 50.0, 100.0])
        target = Vector3([0.0, 0.0, 0.0])
        up = Vector3([0.0, 1.0, 0.0])
        view = Matrix44.look_at(eye, target, up)
        proj = Matrix44.perspective_projection(70.0, self.wnd.aspect_ratio, 0.1, 1000.0)
        mvp = proj * view * self.model

        self.prog["mvp"].write(mvp.astype("f4").tobytes())

        # Compute the normal matrix (inverse-transpose of model's upper 3x3)
        normal_matrix = np.linalg.inv(self.model[:3, :3]).T
        self.prog["normal_matrix"].write(normal_matrix.astype("f4").tobytes())

        self.vao.render()


if __name__ == "__main__":
    mglw.run_window_config(HeightfieldApp)
