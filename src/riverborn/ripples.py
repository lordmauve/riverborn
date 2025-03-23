"""
Water Ripple Simulation Demo using ModernGL and moderngl-window.
Run with: python water_ripple.py
Dependencies: ModernGL, moderngl-window, numpy
"""

import moderngl
import moderngl_window
from moderngl_window import geometry
import numpy as np


VERTEX_SHADER = """
#version 330
in vec2 in_position;
out vec2 v_texcoord;
void main() {
    // in_position is in the range [-1,1] for a full-screen quad
    v_texcoord = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

# The simulation shader computes the new height value from the neighbors of the current height texture
# and subtracts the previous height (thus implicitly storing velocity) then damps the result.
SIMULATION_FRAGMENT_SHADER = """
#version 330
uniform sampler2D curr_tex;
uniform vec2 texel;

in vec2 v_texcoord;
out vec4 fragColor;

const float dt = 1.0;
const float tension = 0.1;
const float damping_factor = 0.99;

void main() {
    float hL = texture(curr_tex, v_texcoord - vec2(texel.x, 0.0)).r;
    float hR = texture(curr_tex, v_texcoord + vec2(texel.x, 0.0)).r;
    float hD = texture(curr_tex, v_texcoord - vec2(0.0, texel.y)).r;
    float hU = texture(curr_tex, v_texcoord + vec2(0.0, texel.y)).r;
    vec2 here = texture(curr_tex, v_texcoord).rg;

    float laplacian = hL + hR + hD + hU - 4.0 * here.r;
    float velocity = here.g + tension * laplacian * dt;
    velocity *= damping_factor;
    float new_height = here.r + velocity * dt;

    fragColor = vec4(new_height, velocity, 0.0, 1.0);
}
"""

# The render shader computes normals by finite differences of the height field and lights the surface.
RENDER_FRAGMENT_SHADER = """
#version 330
uniform sampler2D height_tex;
uniform vec2 texel;
uniform vec3 light_dir;  // direction of the light (e.g. from top-left)
in vec2 v_texcoord;
out vec4 fragColor;
void main() {
    float hL = texture(height_tex, v_texcoord - vec2(texel.x, 0.0)).r;
    float hR = texture(height_tex, v_texcoord + vec2(texel.x, 0.0)).r;
    float hD = texture(height_tex, v_texcoord - vec2(0.0, texel.y)).r;
    float hU = texture(height_tex, v_texcoord + vec2(0.0, texel.y)).r;
    // Compute the approximate normal: the z component is set to 1.0 for scale.
    vec3 normal = normalize(vec3(hL - hR, hD - hU, 1.0));
    float diff = max(dot(normal, normalize(light_dir)), 0.0);
    // Base water color modulated by diffuse lighting.
    vec3 baseColor = vec3(0.0, 0.5, 1.0);
    fragColor = vec4(baseColor * diff, 1.0);
}
"""

# The disturbance shader “draws” a line into the height buffer.
# It takes two endpoints (p1, p2) in texture coordinates and adds a disturbance whose strength falls off with distance.
DISTURBANCE_FRAGMENT_SHADER = """
#version 330
uniform vec2 p1;         // start point of line in texture coords
uniform vec2 p2;         // end point of line in texture coords
uniform float thickness; // effective thickness of the line
uniform float intensity; // disturbance strength to add
in vec2 v_texcoord;
out vec4 fragColor;
void main() {
    // Compute distance from the current fragment to the line segment p1-p2.
    vec2 pa = v_texcoord - p1;
    vec2 ba = p2 - p1;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    float dist = length(pa - ba * h);
    // Smoothly add an impulse if within the thickness.
    float effect = (1.0 - smoothstep(0.0, thickness, dist)) * intensity;
    fragColor = vec4(effect, 0.0, 0.0, 1.0);
}
"""


class WaterRippleDemo(moderngl_window.WindowConfig):
    gl_version = (3, 3)
    title = "GPU Water Ripple Simulation"
    window_size = (800, 600)
    aspect_ratio = None  # Let the window determine the aspect ratio.
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fbos = []
        self.height_textures = []

        # Use buffer index 0 as current, 1 as previous
        self.current_idx = 0
        self.prev_idx = 1

        # Create the full-screen quad (with vertices in clip space)
        self.quad = geometry.quad_fs()

        # Compile the simulation program
        self.sim_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=SIMULATION_FRAGMENT_SHADER,
        )
        # Compile the render program (to display the water surface with lighting)
        self.render_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=RENDER_FRAGMENT_SHADER,
        )
        # Compile the disturbance program (for mouse drawing)
        self.disturb_prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=DISTURBANCE_FRAGMENT_SHADER,
        )

        self.on_resize(self.wnd.size[0], self.wnd.size[1])

        # Set uniforms that remain constant
        texel = (1.0 / self.sim_width, 1.0 / self.sim_height)
        self.sim_prog['texel'].value = texel
        self.render_prog['texel'].value = texel
        # Light coming from top-left (adjust as desired)
        self.render_prog['light_dir'].value = (-0.5, -0.5, 1.0)

        # For disturbance, set default thickness and intensity (tweak these)
        self.disturb_prog['thickness'].value = 0.005
        self.disturb_prog['intensity'].value = 0.5

        # For mouse drawing: store last mouse position in texture coordinates (or None)
        self.last_mouse = None

    def on_render(self, time, frame_time):
        # First, run the simulation update.
        # We will render the new state into the FBO that is not currently the "current" texture.
        target_fbo = self.fbos[self.prev_idx]
        target_fbo.use()

        # Bind the current and previous height textures to texture units 0 and 1 respectively.
        self.height_textures[self.current_idx].use(location=0)
        self.sim_prog['curr_tex'].value = 0

        # Render full-screen quad to update the simulation
        self.quad.render(self.sim_prog)

        # Swap the buffers: current becomes previous, and new becomes current.
        self.prev_idx, self.current_idx = self.current_idx, self.prev_idx

        # Now render the water surface to the screen.
        self.ctx.screen.use()
        self.ctx.screen.clear(0.0, 0.0, 0.0, 1.0)
        self.height_textures[self.current_idx].use(location=0)
        self.render_prog['height_tex'].value = 0
        self.quad.render(self.render_prog)

    def on_mouse_drag_event(self, x, y, dx, dy):
        # Convert mouse coordinates (window: origin top-left) to texture coordinates (origin bottom-left)
        width, height = self.wnd.size
        cur_pos = (x / width, 1 - y / height)
        if self.last_mouse is None:
            self.last_mouse = cur_pos

        # Draw the disturbance into the current simulation texture.
        # We do this by rendering the disturbance quad into the current FBO with additive blending.
        self.fbos[self.current_idx].use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        self.disturb_prog['p1'].value = self.last_mouse
        self.disturb_prog['p2'].value = cur_pos
        self.quad.render(self.disturb_prog)
        self.ctx.disable(moderngl.BLEND)
        self.last_mouse = cur_pos

    def on_mouse_press_event(self, x, y, button):
        # Start drawing: record the initial mouse position in texture coords.
        width, height = self.wnd.size
        self.last_mouse = (x / width, 1 - y / height)

    def on_mouse_release_event(self, x, y, button):
        self.last_mouse = None

    def on_resize(self, width: int, height: int):
        # When the window is resized, update simulation textures and uniforms.
        self.sim_width, self.sim_height = width, height
        texel = (1.0 / width, 1.0 / height)
        self.sim_prog['texel'].value = texel
        self.render_prog['texel'].value = texel

        # Create the simulation textures and FBOs at the new resolution.
        for i in range(len(self.height_textures)):
            # Release the old textures and FBOs
            self.height_textures[i].release()
            self.fbos[i].release()
        self.fbos = []
        self.height_textures = []
        for i in range(2):
            tex = self.ctx.texture((width, height), components=2, dtype='f4')
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            fbo.clear(color=(0.0, 0.0, 0.0, 1.0))
            self.height_textures.append(tex)
            self.fbos.append(fbo)


if __name__ == '__main__':
    moderngl_window.run_window_config(WaterRippleDemo)
