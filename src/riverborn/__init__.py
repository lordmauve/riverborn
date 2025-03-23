from functools import partial
import time
from ._version import __version__

import numpy as np
import noise
import pygfx as gfx
import pylinalg as la

# -----------------------------
# Parameters for the heightfield
# -----------------------------
width = 100            # overall width of the plane
height = 100           # overall height (depth) of the plane
segments = 100         # number of subdivisions along each axis
noise_scale = 0.05      # scale factor for noise sampling
height_multiplier = 10  # multiply noise value to exaggerate heights

# -----------------------------
# Create the plane geometry
# -----------------------------
# This creates a grid with vertices we can adjust.
plane = gfx.geometries.plane_geometry(
    width=width,
    height=height,
    width_segments=segments,
    height_segments=segments,
)

perlin = partial(noise.pnoise2, octaves=4, persistence=0.5, lacunarity=2.0,
                    repeatx=1024, repeaty=1024, base=42)

# Modify the vertex positions to create a heightfield.
# The positions attribute is an (N, 3) numpy array.
positions = plane.positions.data
normals = plane.normals.data
for i, pos in enumerate(positions):
    x, y, z = pos
    assert z == 0
    # Sample Perlin noise for the (x, y) coordinate.
    n = perlin(x * noise_scale, y * noise_scale)

    # Compute the normal
    e = 0.2
    dndx = (perlin((x + e) * noise_scale, y * noise_scale) - n) / e
    dndy = (perlin(x * noise_scale, (y + e) * noise_scale) - n) / e
    normals[i] = -dndx, e, -dndy
    # Set the height based on the noise value.
    # The height is exaggerated by a multiplier for better visibility.
    pos[:] = x, n * height_multiplier, -y

la.vec_normalize(normals)
print("Generatd heightfield")

# -----------------------------
# Create a texture using Perlin noise
# -----------------------------
# Here we generate a simple grayscale image where the intensity comes from noise.
tex_width, tex_height = 256, 256
texture_data = np.zeros((tex_height, tex_width, 3), dtype=np.uint8)
texture_noise_scale = 0.1

for i in range(tex_height):
    for j in range(tex_width):
        # Sample noise; note the base value is different for variety.
        t = noise.pnoise2(j * texture_noise_scale, i * texture_noise_scale,
                           octaves=4, persistence=0.5, lacunarity=2.0,
                           repeatx=1024, repeaty=1024, base=24)
        # Normalize the value from [-1,1] to [55,255]
        color = int((t + 1) * 0.5 * 200 + 55)
        texture_data[i, j] = (color, color, color)

# Create a pygfx texture object.
texture = gfx.Texture(texture_data, dim=2)

# -----------------------------
# Create a material and mesh
# -----------------------------
# Use a Phong material to benefit from lighting and add our texture.
#material = gfx.MeshPhongMaterial(color='#888888')
material = gfx.MeshStandardMaterial(
    color=(0.6, 0.5, 0.4),
    roughness=5,
    metalness=0.1,
    map=texture,
)
mesh = gfx.Mesh(plane, material)

# -----------------------------
# Set up the scene, camera, and renderer
# -----------------------------
scene = gfx.Scene()
scene.add(mesh)

water = gfx.Mesh(
    gfx.geometries.plane_geometry(
        width=width,
        height=height,
    ),
    gfx.MeshPhongMaterial(color='#8888ff20', specular='#fff8ee', shininess=50),
)
water.local.rotation = (0.7071, 0, 0, 0.7071)
scene.add(water)

# Add a light
scene.add(light := gfx.DirectionalLight(color=(1, 1, 1), intensity=1))
light.local.position = (10, 10, 10)

camera = gfx.PerspectiveCamera(70, 16/9)
camera.local.position = (50, 5, 0)
camera.show_pos((0, 0, 0))
scene.add(camera)

# Optional: Add orbit controls for interactive rotation
controls = gfx.OrbitController(camera)


rot = la.quat_from_euler((0, 0.01, 0.0), order="XYZ")


def animate():
    # Slowly rotate the mesh for a dynamic view.
    #mesh.local.rotation = la.quat_mul(rot, mesh.local.rotation)
    t = time.monotonic()


    camera.local.position = (20 * np.cos(0.1 * t), 10, 20 * np.sin(0.1 * t))
    camera.show_pos((0, 0, 0))

def main():
    gfx.show(
        scene,
        #controller=controls,
        before_render=animate,
    )
