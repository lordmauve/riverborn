"""Shadow mapping system that integrates with the Scene framework."""
import moderngl
import moderngl_window as mglw
import numpy as np
from pyglm import glm
from riverborn.camera import Camera
from riverborn.scene import Scene, Model, WavefrontModel, TerrainModel, Light, Material
from riverborn.shader import load_shader, BindableProgram


class ShadowMap:
    """Shadow map implementation.

    Attributes:
        width: Width of the shadow map
        height: Height of the shadow map
        depth_texture: Depth texture for shadow mapping
        fbo: Framebuffer for rendering to the depth texture
        depth_shader: Shader program for depth pass
    """
    def __init__(self, width: int = 2048, height: int = 2048):
        """Initialize the shadow map.

        Args:
            width: Width of the shadow map texture
            height: Height of the shadow map texture
        """
        self.ctx = mglw.ctx()
        self.width = width
        self.height = height

        # Create a depth texture
        self.depth_texture = self.ctx.depth_texture((width, height))
        self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Create a framebuffer with the depth texture
        self.fbo = self.ctx.framebuffer(depth_attachment=self.depth_texture)


class ShadowSystem:
    """System for rendering shadows using shadow mapping.

    This class handles the shadow mapping process for a scene, rendering
    the scene from the light's perspective into a shadow map, then using
    that shadow map when rendering the scene normally to add shadows.
    """
    def __init__(self, light: Light, shadow_map_size: int = 2048, use_pcf: bool = True):
        """Initialize the shadow system.

        Args:
            shadow_map_size: Size of the shadow map texture (width & height)
            use_pcf: Whether to use percentage closer filtering for smoother shadows
        """
        self.shadow_map = ShadowMap(shadow_map_size, shadow_map_size)
        self.light = light
        self.use_pcf = use_pcf

    def get_shadow_shader(self, instanced=True, material=None) -> BindableProgram:
        """Get an appropriate shadow shader based on material properties.

        Args:
            instanced: Whether to use instancing
            material: Material properties to pass to the shader

        Returns:
            Shader program for shadow rendering
        """
        # Create shader defines based on material properties
        defines = {}

        if instanced:
            defines['INSTANCED'] = '1'

        if material:
            defines.update(material.to_defines())

        return load_shader('shadow', **defines)

    def render_depth(self, scene: Scene):
        """Render the scene to the shadow map from the light's perspective.

        Args:
            scene: The scene to render
        """
        if not self.light:
            raise ValueError("Light not set")

        # Bind shadow map framebuffer and clear it
        self.shadow_map.fbo.clear(depth=1.0)
        self.shadow_map.fbo.use()

        # Enable depth testing
        ctx = mglw.ctx()
        ctx.enable(moderngl.DEPTH_TEST)

        # Set viewport to shadow map size
        previous_viewport = ctx.viewport
        ctx.viewport = (0, 0, self.shadow_map.width, self.shadow_map.height)
        scene.render_depth(self.light.light_space_matrix)
        ctx.viewport = previous_viewport
        ctx.screen.use()

    def setup_shadow_shader(self, camera: Camera, model: Model, **uniforms):
        """Set up the shadow shader for a specific model."""
        # Choose the appropriate shader for the model
        shader = self.get_shadow_shader(instanced=True, material=model.material)
        shader.bind(
            m_view=camera.get_view_matrix(),
            m_proj=camera.get_proj_matrix(),
            light_dir=-self.light.direction,
            light_color=self.light.color,
            ambient_color=self.light.ambient,
            camera_pos=camera.eye,
            light_space_matrix=self.light.light_space_matrix,
            shadow_map=self.shadow_map.depth_texture,
            use_pcf=self.use_pcf,
            pcf_radius=1.0,
            shadow_bias=0.01,
            **uniforms
        )
        return shader
