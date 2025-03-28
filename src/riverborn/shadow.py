"""Shadow mapping system that integrates with the Scene framework."""
import moderngl
import moderngl_window as mglw
import numpy as np
from pyglm import glm

from riverborn.camera import Camera
from riverborn.scene import Scene, Model, WavefrontModel, TerrainModel, Light
from riverborn.shader import load_shader



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
        #self.depth_texture.compare_func = '<='
        self.depth_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Create a framebuffer with the depth texture
        self.fbo = self.ctx.framebuffer(depth_attachment=self.depth_texture)

        # Load depth shader program
        self.depth_shader_instanced = load_shader('depth', defines={'INSTANCED': '1', 'ALPHA_TEST': '1'})
        self.depth_shader_uniform = load_shader('depth', defines={'ALPHA_TEST': '1'})


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
        self.shadow_shader_instanced = load_shader('shadow', defines={'INSTANCED': '1'})
        self.shadow_shader_uniform = load_shader('shadow')
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

        # Render each model in the scene
        for model_name, model in scene.models.items():
            # If instances_dirty is set, update the instance buffer
            model.flush_instances()

            # Render each part of the model
            for part in model.parts:
                # Create VAO specifically for the depth pass
                # This creates a temporary VAO just for the depth pass
                # The structure depends on the vertex format from the model's parts

                # Update light space matrix uniform for the depth shader
                depth_shader = self.shadow_map.depth_shader_instanced
                depth_shader.bind(
                    light_space_matrix=self.light.light_space_matrix,
                    **part['uniforms']
                )

                # Find the right vertex format based on model type
                if isinstance(model, WavefrontModel):
                    vertex_format = '2f 3x4 3f'  # texcoord, normal, position
                    attrs = 'in_texcoord_0', 'in_position'
                elif isinstance(model, TerrainModel):
                    vertex_format = '3f 3x4 2f'  # position, normal, texcoord
                    attrs = 'in_position', 'in_texcoord_0'
                else:
                    # Default case - attempt to use just position component
                    # This will need to be adjusted for other model types
                    vertex_format = '3f'  # position only
                    attrs = 'in_position'

                # Extract just the position component for depth pass
                vao_args = [
                    (part['vbo'], vertex_format, *attrs),
                    (model.instance_buffer, '16f4/i', 'm_model')
                ]

                # Create the VAO (with or without indices)
                if 'ibo' in part:
                    vao = ctx.vertex_array(depth_shader, vao_args, part['ibo'])
                else:
                    vao = ctx.vertex_array(depth_shader, vao_args)

                # Render this part with instancing
                print(f"Rendering {model.instance_count}")
                vao.render(instances=model.instance_count)

                # Clean up the temporary VAO
                vao.release()

        # Restore viewport
        ctx.viewport = previous_viewport

    def setup_shadow_shader(self, camera: Camera, model: Model, **uniforms):
        """Set up the shadow shader for a specific model."""
        # Choose the appropriate shader for the model
        shader = self.shadow_shader_instanced

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

    def render_scene_with_shadows(self, scene: Scene, camera: Camera):
        """Render a complete scene with shadows applied.

        Args:
            scene: The scene to render
            camera: The camera to use for rendering
        """
        # First render the depth pass from light's perspective
        self.render_depth(scene)

        # Setup viewport for main rendering
        ctx = mglw.ctx()
        ctx.enable(moderngl.DEPTH_TEST)

        # Now render the scene with shadows
        for model_name, model in scene.models.items():
            # Set up the shadow shader for this model
            shader = self.setup_shadow_shader(camera, model)

            # Temporarily override the model's program
            original_program = model.program
            model.program = shader

            # Render the model with our shadow shader
            model.draw(camera, -self.light.direction)

            # Restore the original program
            model.program = original_program
