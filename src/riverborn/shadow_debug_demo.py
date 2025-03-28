"""Standalone shadow map debugging demo."""
import moderngl
import moderngl_window as mglw
import numpy as np
from pyglm import glm
import math

from riverborn.camera import Camera
from riverborn.shadow import ShadowSystem, Light
from riverborn.shader import load_shader
from riverborn.terrain import make_terrain
from riverborn.heightfield import create_noise_texture, Instance as TerrainInstance
from riverborn.shadow_debug import render_shadow_map_to_screen, render_small_shadow_map


class ShadowDebugDemo(mglw.WindowConfig):
    """Demo that visualizes the shadow map directly."""
    gl_version = (3, 3)
    title = "Shadow Map Debug"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)

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

        # Create the terrain
        # Note where the terrain's center is located
        terrain_mesh = make_terrain(100, 50, 50, 8, 0.05)
        self.terrain = TerrainInstance(
            terrain_mesh,
            load_shader('shadow'),
            create_noise_texture(1024, color=(0.6, 0.5, 0.4))
        )

        # Create a directional light - now with target at the terrain center
        terrain_center = glm.vec3(0.0, 0.0, 0.0)  # Adjust if your terrain isn't centered at origin
        self.light = Light(
            direction=[0.5, -0.8, -0.3],
            color=[1.0, 1.0, 1.0],
            ambient=[0.2, 0.2, 0.2],
            ortho_size=60.0,  # Increased size to ensure terrain is covered
            near=1.0,
            far=200.0,
            target=terrain_center  # Target the center of the terrain
        )

        # Create the shadow system
        self.shadow_system = ShadowSystem(self.light, shadow_map_size=1024)

        # Time tracking
        self.time = 0
        self.rotate_light = True

        # Debug options
        self.view_full_screen = False
        self.use_checkerboard = True
        self.display_mode = 0  # 0: raw, 1: linearized, 2: test pattern
        self.display_modes = ["Raw Depth", "Linearized Depth", "Test Pattern"]

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
                -0.2,
                math.cos(angle)
            ))
            self.light.update_matrices()

        # We'll skip the regular render_depth call since we don't have a scene
        # Instead, manually render the terrain to the shadow map

        # Bind shadow map framebuffer and clear it
        self.shadow_system.shadow_map.fbo.clear(depth=1.0)
        self.shadow_system.shadow_map.fbo.use()

        # Enable depth testing
        ctx = mglw.ctx()
        ctx.enable(moderngl.DEPTH_TEST)

        # Set viewport to shadow map size
        previous_viewport = ctx.viewport
        ctx.viewport = (0, 0, self.shadow_system.shadow_map.width, self.shadow_system.shadow_map.height)

        # Get depth shader for terrain rendering
        depth_shader = self.shadow_system.shadow_map.depth_shader_uniform

        light_space_matrix = self.light.light_space_matrix
        # Set all uniforms at once including m_model
        depth_shader.bind(
            light_space_matrix=light_space_matrix,
            m_model=self.terrain.matrix
        )

        # Create VAO for terrain with model matrix attribute
        terrain_depth_vao = ctx.vertex_array(
            depth_shader,
            [
                (self.terrain.vbo, '3f4 5x4', 'in_position')  # Access vbo from terrain instance
            ],
            self.terrain.ibo  # Access ibo from terrain instance
        )

        # Render terrain to depth buffer
        terrain_depth_vao.render()

        # Restore viewport
        ctx.viewport = previous_viewport

        # Now visualize the shadow map
        if self.view_full_screen:
            # Full screen visualization
            ctx.screen.use()
            render_shadow_map_to_screen(
                self.shadow_system.shadow_map.depth_texture,
                near_plane=self.light.near,
                far_plane=self.light.far
            )
        else:
            # Render normal scene with shadows
            ctx.screen.use()

            # Render terrain with shadows
            self.shadow_system.setup_shadow_shader(
                self.camera,
                self.terrain.prog,
                texture0=self.terrain.texture,
                m_model=self.terrain.matrix
            )
            self.terrain.vao.render()

            if self.debug_mode:
                render_small_shadow_map(
                    *self.wnd.buffer_size,
                    self.shadow_system,
                    self.light
                )

    debug_mode = True
    recorder = None

    def on_key_event(self, key, action, modifiers):
        op = 'press' if action == self.wnd.keys.ACTION_PRESS else 'release'
        keys = self.wnd.keys
        match op, key, modifiers.shift:
            case ('press', keys.ESCAPE, _):
                sys.exit()

            case ('press', keys.F12, False):
                from riverborn.screenshot import screenshot
                screenshot()

            case ('press', keys.F12, True):
                if self.recorder is None:
                    from riverborn.screenshot import VideoRecorder
                    self.recorder = VideoRecorder()
                self.recorder.toggle_recording()

            case ('press', keys.SPACE, _):
                self.rotate_light = not self.rotate_light
                print(f"Light rotation: {'on' if self.rotate_light else 'off'}")
            case ('press', keys.F, _):
                self.view_full_screen = not self.view_full_screen
                print(f"Full screen mode: {'on' if self.view_full_screen else 'off'}")
            case ('press', keys.D, _):
                self.display_mode = (self.display_mode + 1) % len(self.display_modes)
                print(f"Display mode: {self.display_modes[self.display_mode]}")
            case ('press', keys.P, _):
                # Print debug info
                print(f"Light direction: {self.light.direction}")
                print(f"Light position: {self.light.position}")
                print(f"Light target: {self.light.target}")
                print(f"Light near/far: {self.light.near}/{self.light.far}")
                print(f"Shadow map size: {self.shadow_system.shadow_map.width}x{self.shadow_system.shadow_map.height}")
            case ('press', keys.C, _):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'on' if self.debug_mode else 'off'}")

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
    """Run the shadow map debug demo."""
    mglw.run_window_config(ShadowDebugDemo)


if __name__ == '__main__':
    main()
