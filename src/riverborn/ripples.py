from contextlib import contextmanager
from typing import Generator
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
import moderngl_window
import numpy as np

from riverborn.blending import blend_func

from .shader import load_shader


class WaterSimulation:
    """
    Encapsulates the water ripple simulation state.
    Initializes simulation textures, framebuffers, shaders, and provides methods
    to update the simulation, render the water surface, and apply disturbances.
    """
    def __init__(self, width, height):
        self.ctx = mglw.ctx()
        self.quad = geometry.quad_fs()
        self.width = width
        self.height = height
        self.current_idx = 0  # index for the "current" texture
        self.prev_idx = 1     # index for the "previous" texture

        # Load simulation, render, and disturbance shaders.
        self.sim_prog = load_shader("ripple", frag="ripple_sim")
        self.render_prog = load_shader("ripple", frag="ripple_render")
        self.disturb_prog = load_shader("ripple", frag="ripple_disturb")

        # Set uniforms that remain constant.
        self.render_prog["light_dir"].value = (-0.5, -0.5, 1.0)
        self.disturb_prog["thickness"].value = 0.005
        self.disturb_prog["intensity"].value = 0.5

        # Create the simulation textures and framebuffers.
        self.fbos = []
        self.height_textures = []
        self.resize(width, height)

    def resize(self, width: int, height: int):
        """
        Re-create simulation textures and FBOs at the new resolution and update uniforms.
        """
        self.width = width
        self.height = height
        texel = (1.0 / width, 1.0 / height)
        self.sim_prog["texel"].value = texel
        self.render_prog["texel"].value = texel

        # Release old textures and FBOs if they exist.
        for tex in self.height_textures:
            tex.release()
        for fbo in self.fbos:
            fbo.release()

        self.fbos = []
        self.height_textures = []
        for i in range(2):
            tex = self.ctx.texture((width, height), components=2, dtype="f4")
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            fbo.clear(color=(0.0, 0.0, 0.0, 1.0))
            self.height_textures.append(tex)
            self.fbos.append(fbo)

    def simulate(self):
        """
        Update the simulation by rendering the new water state into the off-screen FBO.
        Uses a scope so that binding the framebuffer and textures is automatically restored.
        """
        target_fbo = self.fbos[self.prev_idx]
        with self.ctx.scope(framebuffer=target_fbo):
            # Bind the current height texture to texture unit 0.
            self.height_textures[self.current_idx].use(location=0)
            self.sim_prog["curr_tex"].value = 0
            # Render a full-screen quad to update the simulation.
            self.quad.render(self.sim_prog)
        # Swap current and previous indices.
        self.current_idx, self.prev_idx = self.prev_idx, self.current_idx

    @property
    def texture(self) -> moderngl.Texture:
        """Get the current height texture."""
        return self.height_textures[self.current_idx]

    def render(self):
        """
        Render the water surface to the active framebuffer (usually the screen).
        Uses a scope for consistency.
        """
        with self.ctx.scope():
            self.texture.use(location=0)
            self.render_prog["height_tex"].value = 0
            self.quad.render(self.render_prog)

    def disturb(self, p1, p2):
        """
        Apply a disturbance (ripple) into the current simulation texture.
        Expects p1 and p2 to be texture coordinates (tuples).
        Uses a scope to temporarily enable blending with additive mode.
        """
        target_fbo = self.fbos[self.current_idx]
        with self.ctx.scope(framebuffer=target_fbo, enable_only=moderngl.BLEND):
            # Set blend function to additive blending (restored after the scope).
            with blend_func(moderngl.SRC_ALPHA, moderngl.ONE):
                self.disturb_prog["p1"].value = p1
                self.disturb_prog["p2"].value = p2
                self.quad.render(self.disturb_prog)


class WaterRippleDemo(mglw.WindowConfig):
    """
    Demo window that uses WaterSimulation to perform a GPU water ripple simulation.
    """
    gl_version = (3, 3)
    title = "GPU Water Ripple Simulation"
    window_size = (1024, 1024)
    aspect_ratio = None  # Let the window determine the aspect ratio.
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create a full-screen quad.
        # Create the WaterSimulation instance.
        self.water_sim = WaterSimulation(self.wnd.size[0], self.wnd.size[1])
        # For mouse drawing: store last mouse position in texture coordinates (or None)
        self.last_mouse = None

    def on_render(self, time, frame_time):
        # Update the water simulation.
        self.water_sim.simulate()

        # Render the water surface to the screen.
        self.ctx.screen.use()
        self.ctx.screen.clear(0.0, 0.0, 0.0, 1.0)
        self.water_sim.render()

    def on_mouse_drag_event(self, x, y, dx, dy):
        # Convert mouse coordinates (window: origin top-left) to texture coordinates (origin bottom-left)
        width, height = self.wnd.size
        cur_pos = (x / width, 1 - y / height)
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

    def on_resize(self, width: int, height: int):
        # Update the water simulation on window resize.
        self.water_sim.resize(width, height)


def main():
    mglw.run_window_config(WaterRippleDemo)
