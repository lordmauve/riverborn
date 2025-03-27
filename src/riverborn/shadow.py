"""Shadow mapping system that integrates with the Scene framework."""
import moderngl
import moderngl_window as mglw
import numpy as np
from pyglm import glm

from riverborn.camera import Camera
from riverborn.scene import Scene, Model
from riverborn.shader import load_shader


class Light:
    """Directional light with shadow mapping capabilities.

    Attributes:
        direction: Direction of the light (will be normalized)
        color: Color of the light
        ambient: Ambient color of the light
        position: Computed position based on direction and distance
        view_matrix: Light's view matrix
        proj_matrix: Light's projection matrix
        light_space_matrix: Combined projection and view matrix
    """
    def __init__(self,
                 direction: glm.vec3,
                 color: glm.vec3 = glm.vec3(1.0),
                 ambient: glm.vec3 = glm.vec3(0.1),
                 distance: float = 100.0,
                 ortho_size: float = 50.0,
                 near: float = 1.0,
                 far: float = 200.0):
        """Initialize the light.

        Args:
            direction: Direction of the light
            color: Color of the light
            ambient: Ambient color in shadow areas
            distance: Distance at which to place the light
            ortho_size: Size of the orthographic projection box
            near: Near plane distance
            far: Far plane distance
        """
        self.direction = glm.normalize(glm.vec3(direction))
        self.color = glm.vec3(color)
        self.ambient = glm.vec3(ambient)
        self.distance = distance
        self.ortho_size = ortho_size
        self.near = near
        self.far = far

        # Position the light opposite to its direction
        self.position = -self.direction * distance
        self.update_matrices()

    def update_matrices(self):
        """Update view and projection matrices."""
        # Create view matrix looking from light position toward the origin
        self.view_matrix = glm.lookAt(
            self.position,                # eye position
            glm.vec3(0.0, 0.0, 0.0),      # looking at origin
            glm.vec3(0.0, 1.0, 0.0)       # up vector
        )

        # Create orthographic projection matrix
        size = self.ortho_size
        self.proj_matrix = glm.ortho(
            -size, size, -size, size, self.near, self.far
        )

        # Combined light space matrix
        self.light_space_matrix = self.proj_matrix * self.view_matrix


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
        self.depth_texture.compare_func = '<='
        self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Create a framebuffer with the depth texture
        self.fbo = self.ctx.framebuffer(depth_attachment=self.depth_texture)

        # Load depth shader program
        self.depth_shader = load_shader('depth', defines={'INSTANCED': '1'})


class ShadowSystem:
    """System for rendering shadows using shadow mapping.

    This class handles the shadow mapping process for a scene, rendering
    the scene from the light's perspective into a shadow map, then using
    that shadow map when rendering the scene normally to add shadows.
    """
    def __init__(self, shadow_map_size: int = 2048, use_pcf: bool = True):
        """Initialize the shadow system.

        Args:
            shadow_map_size: Size of the shadow map texture (width & height)
            use_pcf: Whether to use percentage closer filtering for smoother shadows
        """
        self.shadow_map = ShadowMap(shadow_map_size, shadow_map_size)
        self.light = None
        self.shadow_shader = load_shader('shadow', defines={'INSTANCED': '1'})
        self.use_pcf = use_pcf

    def set_light(self, light: Light):
        """Set the light used for shadow mapping."""
        self.light = light

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

        # Update light space matrix uniform for the depth shader
        depth_shader = self.shadow_map.depth_shader
        depth_shader['light_space_matrix'].write(self.light.light_space_matrix)

        # Render each model in the scene
        for model_name, model in scene.models.items():
            # Create VAOs for each part of the model
            vaos = []
            for part in model.parts:
                # Only need position for depth pass
                vao = ctx.vertex_array(
                    depth_shader,
                    [
                        (part['vbo'], '3f', 'in_position'),
                        (model.instance_buffer, '16f4/i', 'm_model')
                    ],
                    part.get('ibo', None)
                )
                vaos.append(vao)

            # Render all parts with instancing
            for vao in vaos:
                vao.render(instances=model.instance_count)

        # Restore viewport
        ctx.viewport = previous_viewport

    def setup_shadow_shader(self, camera: Camera, shader: moderngl.Program, **uniforms):
        """Set up the shadow shader with common uniforms."""
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
            shadow_bias=0.005,
            **uniforms
        )

    def render_model_with_shadows(self, model: Model):
        """Render a single model with shadows applied.

        Args:
            model: The model to render
        """
        shader = self.shadow_shader
        ctx = mglw.ctx()

        # Update instance buffer if needed
        if model.instances_dirty:
            model.instance_buffer.write(model.instance_matrices[:model.instance_count])
            model.instances_dirty = False

        # Render each part of the model
        for part in model.parts:
            if texture := part.get('texture'):
                texture.use(location=0)
                shader['texture0'].value = 0

            # Create VAO for this part
            vao = ctx.vertex_array(
                shader,
                [
                    (part['vbo'], '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0'),
                    (model.instance_buffer, '16f4/i', 'm_model')
                ],
                part.get('ibo', None)
            )

            # Render all instances
            vao.render(instances=model.instance_count)
