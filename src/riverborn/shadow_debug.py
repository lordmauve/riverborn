"""Module for debugging shadow rendering."""
import moderngl
import moderngl_window as mglw
import numpy as np
from riverborn.shader import load_shader

# Module-level cache for the quad and shader to avoid recreating them each frame
_debug_quad = None
_debug_shader = None

def get_debug_resources():
    """Get or create the debug quad and shader."""
    global _debug_quad, _debug_shader

    if _debug_quad is None or _debug_shader is None:
        ctx = mglw.ctx()
        _debug_shader = load_shader('debug_depth')

        # Quad vertices: x, y, z, u, v
        quad_vertices = np.array([
            # pos(x, y, z), texcoord(u, v)
            -1.0, -1.0, 0.0, 0.0, 0.0,  # bottom-left
            1.0, -1.0, 0.0, 1.0, 0.0,   # bottom-right
            1.0, 1.0, 0.0, 1.0, 1.0,    # top-right
            -1.0, 1.0, 0.0, 0.0, 1.0,   # top-left
        ], dtype='f4')

        # Quad indices
        quad_indices = np.array([
            0, 1, 2,  # triangle 1
            0, 2, 3   # triangle 2
        ], dtype='i4')

        # Create VBO and IBO
        vbo = ctx.buffer(quad_vertices)
        ibo = ctx.buffer(quad_indices)

        # Create VAO with our debug shader
        _debug_quad = ctx.vertex_array(
            _debug_shader,
            [
                (vbo, '3f 2f', 'in_position', 'in_texcoord_0')
            ],
            ibo
        )

    return _debug_quad, _debug_shader

def render_shadow_map_to_screen(shadow_map_texture, near_plane=1.0, far_plane=100.0):
    """Render the shadow map to screen for debugging.

    Args:
        shadow_map_texture: The shadow map depth texture from the shadow pass
        near_plane: Near plane value used for depth linearization
        far_plane: Far plane value used for depth linearization
    """
    # Get the cached quad and shader
    quad, debug_shader = get_debug_resources()

    # Bind the shadow map texture
    shadow_map_texture.use(location=0)

    # Use bind() helper to set all uniforms at once
    debug_shader.bind(
        shadow_map=0,
        near_plane=near_plane,
        far_plane=far_plane
    )

    # Render the quad
    quad.render()
