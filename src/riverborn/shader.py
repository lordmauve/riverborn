from importlib.resources import files

import moderngl


shaders = files() / "shaders"


def load_shader(ctx: moderngl.Context, name: str) -> moderngl.Program:
    return ctx.program(
        vertex_shader=(shaders / f"{name}.vert").read_text(),
        fragment_shader=(shaders / f"{name}.frag").read_text(),
    )
