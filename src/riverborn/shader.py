from importlib.resources import files

import moderngl
import moderngl_window as mglw


shaders = files() / "shaders"


def load_shader(name: str, *, vert: str | None = None, frag: str = None) -> moderngl.Program:
    """Load a shader from the shaders directory."""
    vert = vert or name
    frag = frag or name

    ctx = mglw.ctx()
    return ctx.program(
        vertex_shader=(shaders / f"{vert}.vert").read_text(),
        fragment_shader=(shaders / f"{frag}.frag").read_text(),
    )
