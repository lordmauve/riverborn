from contextlib import contextmanager

import moderngl
from typing import Generator
import moderngl_window


_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)


@contextmanager
def blend_func(src, dest) -> Generator[None, None, None]:
    """
    Context manager to temporarily set the blend function.
    Restores the original blend function after exiting the context.
    """
    global _func
    ctx = moderngl_window.ctx()
    old_func = _func
    _func = ctx.blend_func = (src, dest)
    try:
        yield
    finally:
        _func = ctx.blend_func = old_func

