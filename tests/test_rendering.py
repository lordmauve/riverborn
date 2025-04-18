import os
import numpy as np
import imageio as iio
import pytest
from pathlib import Path
import moderngl
import moderngl_window as mglw
from riverborn import terrain as terrain_mod
from riverborn.screenshot import capture_screen
from riverborn.scene import Scene, Material, Light
from riverborn.camera import Camera
from riverborn.heightfield import create_noise_texture
from riverborn.terrain import make_terrain
from riverborn.shader import load_shader
from riverborn.ripples import WaterSimulation
from riverborn.animals import Animals
from pyglm import glm

REFERENCE_DIR = Path(__file__).parent
TOLERANCE = 5  # mean absolute pixel difference


def setup_scene(ctx, width, height, *, terrain=False, water=False, models=False, shadowing=False):
    mglw.activate_context(ctx=ctx)
    scene = Scene()
    camera = Camera(
        eye=[0.0, 10.0, 20.0],
        target=[0.0, 0.0, 0.0],
        up=[0.0, 1.0, 0.0],
        fov=70.0,
        aspect=width/height,
        near=0.1,
        far=200.0,
    )
    light = Light(
        direction=[0.5, -0.8, -0.3],
        color=[1.0, 0.9, 0.8],
        ambient=[0.2, 0.2, 0.2],
        ortho_size=60.0,
        near=1.0,
        far=200.0,
        target=glm.vec3(0.0, 0.0, 0.0),
        shadows=shadowing
    )
    terrain_instance = None
    if terrain or water or models or shadowing:
        terrain_texture = create_noise_texture(size=256, color=(0.6, 0.5, 0.4))
        terrain_material = Material(receive_shadows=True, cast_shadows=True)
        terrain_model = scene.create_terrain(
            'terrain', segments=64, width=40, depth=40, height=5, noise_scale=0.1,
            texture=terrain_texture, material=terrain_material
        )
        terrain_instance = scene.add(terrain_model)
    if models or shadowing:  # Always load model for shadowing test too
        # Plant material with alpha testing for transparency
        plant_material = Material(receive_shadows=True, cast_shadows=True, alpha_test=True)
        try:
            # Load the plant model with its default texture
            # The texture loading happens automatically in the load_wavefront method
            # when the material specified uses alpha_test=True
            plant_model = scene.load_wavefront('plant13.obj', material=plant_material, capacity=1)
            plant = scene.add(plant_model)
            plant.pos = glm.vec3(0, 0, 0)
            plant.scale = glm.vec3(0.1, 0.1, 0.1)
        except Exception as e:
            print(f"Failed to load plant model: {e}")
    if water:
        # Set up water simulation and geometry
        water_sim = WaterSimulation(256, 256)
        # Water plane geometry: a quad covering the region
        water_size = 40.0
        water_vertices, water_indices = np.array([
            (-water_size, 0.0, -water_size, 0.0, 0.0),
            (water_size, 0.0, -water_size, 1.0, 0.0),
            (water_size, 0.0, water_size, 1.0, 1.0),
            (-water_size, 0.0, water_size, 0.0, 1.0),
        ], dtype='f4'), np.array([0, 1, 2, 0, 2, 3], dtype='i4')
        water_vbo = ctx.buffer(water_vertices.tobytes())
        water_ibo = ctx.buffer(water_indices.tobytes())
        water_prog = load_shader('water')
        water_vao = ctx.vertex_array(
            water_prog,
            [(water_vbo, '3f 2f', 'in_position', 'in_uv')],
            water_ibo
        )
        return scene, camera, light, water_sim, water_prog, water_vao
    return scene, camera, light, None, None, None


def render_scene_headless(width=640, height=480, *, terrain=False, water=False, models=False, shadowing=False):
    ctx = moderngl.create_standalone_context()
    color_tex = ctx.texture((width, height), 3)
    depth_tex = ctx.depth_texture((width, height))
    fbo = ctx.framebuffer(color_attachments=[color_tex], depth_attachment=depth_tex)
    fbo.use()
    ctx.clear(0.2, 0.4, 0.6)
    scene, camera, light, water_sim, water_prog, water_vao = setup_scene(
        ctx, width, height, terrain=terrain, water=water, models=models, shadowing=shadowing
    )

    # Make sure shadow system is properly initialized if shadowing is enabled
    if shadowing and light.shadows:
        shadow_system = light.get_shadow_system()
        # Ensure light matrices are updated
        light.update_matrices()
        # Render the shadow pass first
        shadow_system.render_depth(scene)

    if terrain or models or shadowing:
        scene.draw(camera, light)

    if water and water_vao is not None:
        # Ensure we are rendering to the correct framebuffer
        fbo.use()
        if water_sim:
            water_sim.simulate()
            water_sim.texture.use(location=0)
            water_prog['height_map'].value = 0
        water_prog['env_cube'].value = 1 if 'env_cube' in water_prog else 0
        water_prog['depth_tex'].value = 2 if 'depth_tex' in water_prog else 0
        water_prog['near'].value = 0.1
        water_prog['far'].value = 200.0
        water_prog['resolution'].value = (width, height)
        water_prog['m_model'].write(glm.mat4(1.0))
        water_prog['base_water'] = (0.2, 0.15, 0.1)
        water_prog['water_opaque_depth'] = 3
        camera.bind(water_prog, pos_uniform='camera_pos')
        with ctx.scope(enable_only=moderngl.BLEND):
            water_vao.render()
    img = capture_screen(fbo)
    return img


def assert_render_matches_reference(img, ref_name):
    ref_path = REFERENCE_DIR / ref_name
    diff_path = REFERENCE_DIR / ("diff_" + ref_name)
    if not ref_path.exists():
        iio.imwrite(ref_path, img)
        pytest.fail(f"Reference screenshot did not exist. Saved new reference to {ref_path}. Please review and re-run the test.")
    ref_img = iio.imread(ref_path)
    if img.shape != ref_img.shape:
        pytest.fail(f"Screenshot shape {img.shape} does not match reference {ref_img.shape}.")
    diff = np.abs(img.astype(np.int16) - ref_img.astype(np.int16))
    mean_diff = diff.mean()
    if mean_diff > TOLERANCE:
        combined = np.concatenate([img, ref_img], axis=1)
        iio.imwrite(diff_path, combined)
        pytest.fail(f"Screenshot differs from reference (mean abs diff={mean_diff:.2f} > {TOLERANCE}). See {diff_path} for visual comparison.")


def test_render_terrain():
    img = render_scene_headless(terrain=True)
    assert_render_matches_reference(img, "reference_render_terrain.png")


def test_render_water():
    img = render_scene_headless(terrain=True, water=True)
    assert_render_matches_reference(img, "reference_render_water.png")


def test_render_models():
    img = render_scene_headless(terrain=True, models=True)
    assert_render_matches_reference(img, "reference_render_models.png")


def test_render_shadowing():
    # Use the standard scene rendering function but with shadowing enabled
    img = render_scene_headless(terrain=True, models=True, shadowing=True)
    assert_render_matches_reference(img, "reference_render_shadowing.png")
