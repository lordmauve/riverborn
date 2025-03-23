import pygfx as gfx

from ._version import __version__


cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)

def main():
    gfx.show(cube)
