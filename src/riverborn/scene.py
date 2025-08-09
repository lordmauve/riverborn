"""
Scene system for instanced rendering using ModernGL and PyWavefront.

This module defines a minimal scene management system supporting instanced rendering
of 3D meshes using ModernGL. Each model holds a buffer of instance transformation
matrices. These matrices are used to render multiple instances of the same mesh
with different transforms using a single draw call.

The system supports:
- Loading OBJ + MTL files with PyWavefront
- Creating terrain meshes using Perlin noise
- Creating a persistent instance buffer per model
- Resizing the buffer dynamically with orphan()
- Dirty flagging to only update GPU buffers when needed
- Directional (sunlight) lighting with diffuse and specular shading
"""

from contextlib import suppress
from functools import cache
import importlib.resources
from pathlib import Path
import typing
import imageio
import noise
import pywavefront
import moderngl
import moderngl_window as mglw
import numpy as np
import logging
from pyglm import glm
import random
import math
import weakref
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field, asdict

from riverborn import terrain
from riverborn.camera import Camera
from riverborn.shader import load_shader, BindableProgram
from riverborn.terrain import blank_terrain, make_terrain, Mesh
from pyglm.glm import array


def compute_bounds(positions: np.ndarray) -> tuple[glm.vec3, glm.vec3]:
    """Compute axis aligned bounds from a set of positions."""
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    return glm.vec3(*mins), glm.vec3(*maxs)


def transform_aabb(min_v: glm.vec3, max_v: glm.vec3, m: glm.mat4) -> tuple[glm.vec3, glm.vec3]:
    """Transform an axis aligned bounding box by a matrix."""
    corners = [
        glm.vec3(min_v.x, min_v.y, min_v.z),
        glm.vec3(min_v.x, min_v.y, max_v.z),
        glm.vec3(min_v.x, max_v.y, min_v.z),
        glm.vec3(min_v.x, max_v.y, max_v.z),
        glm.vec3(max_v.x, min_v.y, min_v.z),
        glm.vec3(max_v.x, min_v.y, max_v.z),
        glm.vec3(max_v.x, max_v.y, min_v.z),
        glm.vec3(max_v.x, max_v.y, max_v.z),
    ]
    transformed = [glm.vec3(m * glm.vec4(c, 1.0)) for c in corners]
    tmins = glm.vec3(min(p.x for p in transformed),
                     min(p.y for p in transformed),
                     min(p.z for p in transformed))
    tmaxs = glm.vec3(max(p.x for p in transformed),
                     max(p.y for p in transformed),
                     max(p.z for p in transformed))
    return tmins, tmaxs


def extract_frustum_planes(mat: glm.mat4) -> list[glm.vec4]:
    """Extract normalized frustum planes from a matrix."""
    m = glm.transpose(mat)
    planes = [
        m[3] + m[0],  # left
        m[3] - m[0],  # right
        m[3] + m[1],  # bottom
        m[3] - m[1],  # top
        m[3] + m[2],  # near
        m[3] - m[2],  # far
    ]
    norm = []
    for p in planes:
        n = glm.vec3(p.x, p.y, p.z)
        l = glm.length(n)
        norm.append(p / l)
    return norm


def aabb_in_frustum(min_v: glm.vec3, max_v: glm.vec3, planes: list[glm.vec4]) -> bool:
    """Return True if the AABB intersects the frustum defined by planes."""
    for p in planes:
        px = max_v.x if p.x >= 0 else min_v.x
        py = max_v.y if p.y >= 0 else min_v.y
        pz = max_v.z if p.z >= 0 else min_v.z
        if glm.dot(glm.vec3(p.x, p.y, p.z), glm.vec3(px, py, pz)) + p.w < 0:
            return False
    return True

if typing.TYPE_CHECKING:
    from riverborn.shadow import ShadowSystem


logging.getLogger("pywavefront").level = logging.ERROR  # quiet, you

vec3ish = Union[glm.vec3, tuple[float, float, float], list[float]]

@dataclass
class Part:
    """A part of a model with specific rendering properties.

    Parts are typically used to represent different materials in a model.
    They contain vertex/index buffers and uniform data but not VAOs,
    which are created on-demand during rendering.
    """
    vbo: moderngl.Buffer
    vbotype: Tuple[str, ...]  # Format string and attribute names
    uniforms: Dict[str, Any]
    depth_uniforms: Dict[str, Any]
    ibo: Optional[moderngl.Buffer] = None
    indexed: bool = False
    vao_args: Optional[List[Tuple]] = None

    def __post_init__(self):
        # Ensure vao_args is initialized if not provided
        if self.vao_args is None:
            self.vao_args = []


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
        shadows: Whether this light casts shadows
        shadow_system: Shadow system for this light (created on demand)
    """
    def __init__(self,
                 direction: vec3ish,
                 color: vec3ish = glm.vec3(1.0),
                 ambient: vec3ish = glm.vec3(0.1),
                 distance: float = 100.0,
                 ortho_size: float = 50.0,
                 near: float = 1.0,
                 far: float = 200.0,
                 target: vec3ish = glm.vec3(0.0),
                 shadows: bool = True):
        """Initialize the light.

        Args:
            direction: Direction of the light
            color: Color of the light
            ambient: Ambient color in shadow areas
            distance: Distance at which to place the light
            ortho_size: Size of the orthographic projection box
            near: Near plane distance
            far: Far plane distance
            target: Target point the light looks at (center of the shadow projection)
            shadows: Whether this light casts shadows
        """
        self.direction = glm.normalize(glm.vec3(direction))
        self.color = glm.vec3(color)
        self.ambient = glm.vec3(ambient)
        self.distance = distance
        self.ortho_size = ortho_size
        self.near = near
        self.far = far
        self.target = glm.vec3(target)
        self.shadows = shadows
        self.shadow_system: 'ShadowSystem' | None = None  # Will be created on demand

        self.update_matrices()

    def update_matrices(self):
        """Update view and projection matrices."""
        # Position the light by moving in the opposite direction from the target
        self.position = self.target - self.direction * self.distance

        # Create view matrix looking from light position toward the target
        self.view_matrix = glm.lookAt(
            self.position,             # eye position
            self.target,               # looking at target
            glm.vec3(0.0, 1.0, 0.0)    # up vector
        )

        # Create orthographic projection matrix
        size = self.ortho_size
        self.proj_matrix = glm.ortho(
            -size, size, -size, size, self.near, self.far
        )

        # Combined light space matrix
        self.light_space_matrix = self.proj_matrix * self.view_matrix

    def get_shadow_system(self):
        """Get or create the shadow system for this light."""
        if self.shadows and self.shadow_system is None:
            from riverborn.shadow import ShadowSystem
            self.shadow_system = ShadowSystem(self, shadow_map_size=2048)
        return self.shadow_system


class Model:
    """
    Base class for 3D models with support for instanced rendering.

    Attributes:
        ctx: ModernGL context
        parts: List of model parts (per material) each with its own VAO
        instance_matrices: CPU-side array of transformation matrices
        instance_buffer: GPU buffer for instance matrices
        instance_capacity: Current allocated capacity (resized as needed)
        instance_count: Number of active instances
        instances_dirty: Flag indicating whether GPU buffer needs update
        material: Material properties for this model
    """

    def __init__(self, ctx: moderngl.Context, capacity: int = 100, material: Optional['Material'] = None) -> None:
        """
        Initialize the model by creating instance buffers.

        Args:
            ctx: ModernGL context
            capacity: Initial number of instances to support
            material: Material properties for this model
        """
        self.ctx = ctx
        self.parts = []
        self.instance_count = 0
        self.instance_matrices = array.zeros(capacity, glm.mat4)
        self.instances_dirty = False
        self.instance_buffer = ctx.buffer(reserve=capacity * 16 * 4)
        self.instance_refs: weakref.WeakValueDictionary[int, Instance] = weakref.WeakValueDictionary()
        self.material = material or Material()
        self.textures = {}  # FIXME: share textures between models
        self.bounds_min = glm.vec3(0.0)
        self.bounds_max = glm.vec3(0.0)
        self._visible_matrices = array.zeros(capacity, glm.mat4)
        self._visible_count = 0

    def load_texture(self, path: str) -> moderngl.Texture:
        """Load and create a ModernGL texture from an image file."""
        if path in self.textures:
            return self.textures[path]

        file = importlib.resources.files('riverborn') / f"models/{path}"
        # Read image using imageio
        with file.open('rb') as f:
            image = imageio.imread(f)
            image = np.flipud(image)
        if image.shape[2] == 3:  # Convert RGB to RGBA
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = image
            rgba[..., 3] = 255
            image = rgba

        h, w, depth = image.shape

        # Create ModernGL texture
        texture = self.ctx.texture((w, h), 4, image.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.build_mipmaps()
        self.textures[path] = texture
        return texture

    def create_texture_from_array(self, data: np.ndarray) -> moderngl.Texture:
        """Create a ModernGL texture from a numpy array."""
        h, w, depth = data.shape
        texture = self.ctx.texture((w, h), depth, data.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.build_mipmaps()
        return texture

    def add_instance(self, matrix: glm.mat4, instance=None) -> int:
        """
        Add a new instance matrix to the model, resizing if capacity is exceeded.

        Args:
            matrix: The transformation matrix for the new instance
            instance: The Instance object associated with this matrix (optional)

        Returns:
            Index of the instance in the buffer
        """
        if self.instance_count >= len(self.instance_matrices):
            self.instance_matrices = self.instance_matrices.repeat(2)
            self.instance_buffer.orphan(size=len(self.instance_matrices) * 16 * 4)
            self._visible_matrices = self._visible_matrices.repeat(2)
        index = self.instance_count
        self.instance_matrices[index] = matrix
        self.instance_count += 1
        self.instances_dirty = True

        # Store a weak reference to the instance at this index
        if instance is not None:
            self.instance_refs[index] = instance

        return index

    def update_instance(self, index: int, matrix: glm.mat4) -> None:
        """
        Update the transformation matrix for an existing instance.

        Args:
            index: Index into the instance matrix buffer
            matrix: New transformation matrix
        """
        self.instance_matrices[index] = matrix
        self.instances_dirty = True

    def _get_appropriate_shader(self, light: Light | None, part: Part) -> BindableProgram:
        """Get the appropriate shader based on light and material properties.

        Args:
            light: Light information to determine if shadows are needed

        Returns:
            Compiled shader program appropriate for current rendering state
        """
        from riverborn.shader import load_shader

        light = light or Light(direction=glm.vec3(1, -1, 1))
        # Determine shader type based on shadow state and material properties
        shader_type = 'shadow' if (light.shadows and self.material.receive_shadows) else 'diffuse'

        # For alpha tested materials, use diffuse when not using shadows
        if self.material.alpha_test and shader_type != 'shadow':
            shader_type = 'diffuse'

        # Set up shader defines based on material properties
        defines = {
            'INSTANCED': '1',  # Always use instancing for models
        }
        if 'diffuse_tex' in part.uniforms and 'in_texcoord_0' in part.vbotype:
            defines['ALPHA_TEST'] = '1'
            defines['TEXTURE'] = '1'

        # Add material defines if using a shadow shader
        if shader_type == 'shadow':
            defines.update(self.material.to_defines())

        return load_shader(shader_type, **defines)

    def _get_depth_shader(self, part: Part) -> BindableProgram:
        """Get an appropriate depth shader based on material properties.

        Args:
            instanced: Whether to use instancing
            material: Material properties to pass to the shader

        Returns:
            Shader program for depth rendering
        """
        # Create shader defines based on material properties
        defines = {
            'INSTANCED': '1',  # Always use instancing for models
        }
        if self.material.alpha_test and 'diffuse_tex' in part.uniforms:
            defines['ALPHA_TEST'] = '1'
        return load_shader('depth', **defines)

    def draw(self, camera: Camera, light: Light) -> None:
        """
        Render the model using instanced rendering with appropriate shader.

        Args:
            camera: Camera object
            light: Light object for lighting calculations
        """
        self.flush_instances()
        if hasattr(self, '_current_planes') and self._current_planes is not None:
            self._cull_instances(self._current_planes)
        else:
            self._visible_matrices[:self.instance_count] = self.instance_matrices[:self.instance_count]
            self._visible_count = self.instance_count
        self.instance_buffer.write(self._visible_matrices[:self._visible_count])

        # Get the appropriate shader based on current state

        for part in self.parts:
            shader = self._get_appropriate_shader(light, part)
            shader_name = getattr(shader, 'label', '').lower()

            # Create uniform dict based on shader and rendering state
            uniforms = {
                'm_proj': camera.get_proj_matrix(),
                'm_view': camera.get_view_matrix(),
                **part.uniforms
            }

            # Add lighting uniforms depending on shader
            if 'light_dir' in shader:
                uniforms['light_dir'] = -light.direction
            elif 'sun_dir' in shader:
                uniforms['sun_dir'] = -light.direction

            if 'light_color' in shader:
                uniforms['light_color'] = light.color
            if 'ambient_color' in shader:
                uniforms['ambient_color'] = light.ambient

            # Add shadow-specific uniforms if needed
            if light.shadows and self.material.receive_shadows and 'shadow' in shader_name:
                # Get the shadow system from light if available
                shadow_system = light.get_shadow_system()
                if shadow_system:
                    shadow_map = shadow_system.shadow_map.depth_texture
                    uniforms.update({
                        'camera_pos': camera.eye,
                        'light_space_matrix': light.light_space_matrix,
                        'shadow_map': shadow_map,
                        'use_pcf': shadow_system.use_pcf,
                        'pcf_radius': 1.0,
                        'shadow_bias': 0.002,
                    })

            # Create VAO for rendering based on part properties
            if part.vao_args:
                vao_args = part.vao_args
            else:
                vao_args = [
                    (part.vbo, *part.vbotype),
                    (self.instance_buffer, '16f4/i', 'm_model')
                ]

            # Create a vertex array for this part (with or without indices)
            if part.indexed:
                vao = self.ctx.vertex_array(shader, vao_args, part.ibo)
            else:
                vao = self.ctx.vertex_array(shader, vao_args)

            # Bind all uniforms and render
            shader.bind(**uniforms)
            vao.render(instances=self._visible_count)

            # Clean up the VAO since we don't store it
            vao.release()

    def flush_instances(self):
        if self.instances_dirty:
            self.instance_buffer.write(self.instance_matrices[:self.instance_count])
            self.instances_dirty = False

    def _cull_instances(self, planes: list[glm.vec4]):
        """Cull instances against the provided frustum planes."""
        count = 0
        for i in range(self.instance_count):
            mat = self.instance_matrices[i]
            wmin, wmax = transform_aabb(self.bounds_min, self.bounds_max, mat)
            if aabb_in_frustum(wmin, wmax, planes):
                self._visible_matrices[count] = mat
                count += 1
        self._visible_count = count

    def destroy(self) -> None:
        """
        Clean up GPU resources used by this model.
        """
        for part in self.parts:
            part.vbo.release()
            if part.indexed and part.ibo:
                part.ibo.release()
        self.instance_buffer.release()
        self.parts.clear()


class WavefrontModel(Model):
    """
    A 3D model loaded from an OBJ file with support for instanced rendering.
    """

    def __init__(self, mesh: pywavefront.Wavefront, ctx: moderngl.Context, capacity: int = 100, material: Optional['Material'] = None) -> None:
        """
        Initialize the model by creating buffers for each mesh part and reserving
        space for per-instance data.

        Args:
            mesh: PyWavefront mesh object
            ctx: ModernGL context
            capacity: Initial number of instances to support
            material: Material properties for this model
        """
        super().__init__(ctx, capacity, material)

        for mesh_name, mesh_obj in mesh.meshes.items():
            for obj_material in mesh_obj.materials:
                if not obj_material:
                    continue
                match obj_material.vertex_format:
                    case 'T2F_N3F_V3F':
                        vbotype = '2f 3f 3f', 'in_texcoord_0', 'in_normal', 'in_position'
                    case 'T2F_V3F':
                        vbotype = '2f 3f', 'in_texcoord_0', 'in_position'
                    case 'N3F_V3F':
                        vbotype = '3f 3f', 'in_normal', 'in_position'
                    case _:
                        raise ValueError(f"Unsupported vertex format: {obj_material.vertex_format}")

                vertices = np.array(obj_material.vertices, dtype='f4')
                vbo = ctx.buffer(vertices.tobytes())
                step = {
                    'T2F_N3F_V3F': 8,
                    'T2F_V3F': 5,
                    'N3F_V3F': 6,
                    'V3F': 3,
                }[obj_material.vertex_format]
                positions = vertices.reshape(-1, step)[:, -3:]
                bmin, bmax = compute_bounds(positions)
                if len(self.parts) == 0:
                    self.bounds_min, self.bounds_max = bmin, bmax
                else:
                    self.bounds_min = glm.vec3(min(self.bounds_min.x, bmin.x),
                                               min(self.bounds_min.y, bmin.y),
                                               min(self.bounds_min.z, bmin.z))
                    self.bounds_max = glm.vec3(max(self.bounds_max.x, bmax.x),
                                               max(self.bounds_max.y, bmax.y),
                                               max(self.bounds_max.z, bmax.z))

                # Create uniforms dictionary for textures if present
                uniforms = {}
                uniforms['diffuse_color'] = obj_material.diffuse
                uniforms['specular_exponent'] = obj_material.shininess
                uniforms['specular_color'] = obj_material.specular[:3]
                if obj_material.texture and 'in_texcoord_0' in vbotype:
                    uniforms['diffuse_tex'] = self.load_texture(Path(obj_material.texture.path).name)
                else:
                    vbotype = subformat(vbotype, normal=True)

                # Create a Part with the appropriate configuration
                part = Part(
                    vbo=vbo,
                    vbotype=vbotype,
                    uniforms=uniforms,
                    depth_uniforms={'diffuse_tex': uniforms['diffuse_tex']} if material.alpha_test else {},
                    indexed=hasattr(obj_material, 'faces') and obj_material.faces
                )

                # Add indices if this part is indexed
                if part.indexed:
                    indices = np.array([i for face in obj_material.faces for i in face], dtype='i4')
                    part.ibo = ctx.buffer(indices.tobytes())

                # Set up VAO arguments for rendering
                part.vao_args = [
                    (vbo, *vbotype),
                    (self.instance_buffer, '16f4/i', 'm_model')
                ]

                self.parts.append(part)


class TerrainModel(Model):
    """
    A 3D model created from a terrain mesh with support for instanced rendering.
    """

    def __init__(self, mesh: Mesh, ctx: moderngl.Context, material: Optional['Material'] = None,
                 texture: Optional[Union[moderngl.Texture, np.ndarray]] = None,
                 capacity: int = 1) -> None:
        """
        Initialize the model from a terrain mesh.

        Args:
            mesh: Terrain mesh object
            ctx: ModernGL context
            material: Material properties for this model
            texture: Texture or numpy array to use for the terrain
            capacity: Initial number of instances to support
        """
        super().__init__(ctx, capacity, material)
        self.mesh = mesh

        # Create the VBO
        vbo = ctx.buffer(mesh.vertices.tobytes())

        # Create the IBO
        ibo = ctx.buffer(mesh.indices.astype("i4").tobytes())

        # Process texture if provided
        uniforms = {
            'specular_exponent': 3,
            'specular_color': (0.3, 0.3, 0.3),
        }
        if isinstance(texture, moderngl.Texture):
            uniforms['diffuse_tex'] = texture
        elif isinstance(texture, np.ndarray):
            uniforms['diffuse_tex'] = self.create_texture_from_array(texture)

        vbotype = ('3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')
        self.part = Part(
            vbo=vbo,
            ibo=ibo,
            indexed=True,
            vbotype=vbotype,
            uniforms=uniforms,
            depth_uniforms={'diffuse_tex': uniforms['diffuse_tex']} if material.alpha_test else {},
            vao_args=[(vbo, *vbotype),
                      (self.instance_buffer, '16f4/i', 'm_model')]
        )

        # Add the part
        self.parts.append(self.part)

        positions = mesh.vertices['in_position']
        bmin, bmax = compute_bounds(positions)
        self.bounds_min, self.bounds_max = bmin, bmax

    def update_mesh(self):
        """Update the mesh data in the GPU buffer."""
        # Update vertex buffer with new mesh data
        self.part.vbo.write(self.mesh.vertices.tobytes())


class Instance:
    """
    An instance of a model with its own position, rotation, and scale.

    Attributes:
        model: Reference to the model
        index: Index in the model's instance buffer
    """
    def __init__(self, model: Model) -> None:
        self.model = model
        self._pos = glm.vec3(0, 0, 0)
        self._rot = glm.quat(1, 0, 0, 0)
        self._scale = glm.vec3(1, 1, 1)
        self.index = model.add_instance(self.matrix, self)  # Pass self to model
        self._deleted = False

    @property
    def pos(self) -> glm.vec3:
        return self._pos

    @pos.setter
    def pos(self, value):
        if not isinstance(value, glm.vec3):
            self._pos = glm.vec3(*value)
        else:
            self._pos = value
        self.model.update_instance(self.index, self.matrix)

    @property
    def rot(self) -> glm.quat:
        return self._rot

    @rot.setter
    def rot(self, value):
        if not isinstance(value, glm.quat):
            # Expecting a tuple (w, x, y, z) or similar.
            self._rot = glm.quat(*value)
        else:
            self._rot = value
        self.model.update_instance(self.index, self.matrix)

    @property
    def scale(self) -> glm.vec3:
        return self._scale

    @scale.setter
    def scale(self, value):
        if not isinstance(value, glm.vec3):
            self._scale = glm.vec3(*value)
        else:
            self._scale = value
        self.model.update_instance(self.index, self.matrix)

    @property
    def matrix(self) -> glm.mat4:
        """Compute transformation matrix using translation, rotation, and scale."""
        return glm.scale(
            glm.translate(self._pos) * glm.mat4(self._rot),
            self._scale
        )

    def update(self) -> None:
        """
        Recalculate and upload the transformation matrix to the model's buffer.
        """
        self.model.update_instance(self.index, self.matrix)

    def translate(self, delta) -> None:
        """
        Translate the instance by the given delta.
        Accepts a glm.vec3 or an iterable convertible to glm.vec3.
        """
        if not isinstance(delta, glm.vec3):
            delta = glm.vec3(*delta)
        self.pos = self.pos + delta  # setter is called, which updates the instance

    def rotate(self, angle, axis) -> None:
        """
        Rotate the instance by a given angle (in radians) about the given axis.
        Axis can be a glm.vec3 or any iterable convertible to glm.vec3.
        """
        if not isinstance(axis, glm.vec3):
            axis = glm.vec3(*axis)
        q = glm.angleAxis(angle, axis)
        self.rot = q * self.rot  # setter is called, which updates the instance

    def scale_by(self, factor) -> None:
        """
        Scale the instance by the given factor.
        Factor can be a scalar or an iterable convertible to glm.vec3.
        """
        if isinstance(factor, (int, float)):
            self.scale = self.scale * factor
        else:
            if not isinstance(factor, glm.vec3):
                factor = glm.vec3(*factor)
            self.scale = self.scale * factor

    def delete(self) -> None:
        """
        Delete this instance from its model.
        The last instance's matrix is copied over to this slot and the count is decremented.
        It also updates the index of any instance that was using the last slot.
        """
        # Early return if already deleted
        if self._deleted:
            return

        # Get the index of the last instance
        last_index = self.model.instance_count - 1

        # If we're not deleting the last instance, we need to move the last one to this slot
        if self.index != last_index:
            # Get the instance that uses the last slot
            last_instance = self.model.instance_refs.get(last_index)

            assert last_instance and not last_instance._deleted

            # Update its index to point to the slot we're removing
            last_instance.index = self.index

            # Update the instance reference mapping
            self.model.instance_refs[self.index] = last_instance

            # Copy the last matrix to this slot
            self.model.instance_matrices[self.index] = self.model.instance_matrices[last_index]

        # Remove the reference to the last slot
        del self.model.instance_refs[last_index]

        # Decrement the instance count if it's positive
        self.model.instance_count -= 1

        # Mark as deleted to prevent further operations on this instance
        self._deleted = True

        # Update the instance buffer to reflect changes
        self.model.instances_dirty = True


@dataclass
class Material:
    """Material properties for rendering models with specific shaders."""
    double_sided: bool = False
    translucent: bool = False
    transmissivity: float = 0.0
    receive_shadows: bool = True
    cast_shadows: bool = True
    alpha_test: bool = False

    def to_defines(self) -> Dict[str, str]:
        """Convert material properties to shader defines."""
        defines = {}

        if self.double_sided:
            defines['DOUBLE_SIDED'] = '1'

        if self.translucent:
            defines['TRANSLUCENT'] = '1'
            defines['TRANSMISSIVITY'] = str(self.transmissivity)

        if self.receive_shadows:
            defines['RECEIVE_SHADOWS'] = '1'

        if self.alpha_test:
            defines['ALPHA_TEST'] = '1'

        return defines

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Material':
        """Create a Material instance from a dictionary."""
        # Filter the dict to only include fields that exist in the dataclass
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


class Scene:
    """
    A simple container for models and their instances.

    Supports loading models from the assets package and drawing by model name.
    """
    def __init__(self) -> None:
        self.ctx = mglw.ctx()
        self.models: Dict[str, Model] = {}
        self.instances: List[Instance] = []
        # Default material properties
        self.default_material = Material()
        # Shadow system (created on demand)
        self._shadow_system: ShadowSystem | None = None

    def _get_shader_for_model(self, base_shader: str, material: Optional[Material] = None, **extra_defines) -> BindableProgram:
        """
        Get an appropriate shader for a model based on material properties and rendering requirements.

        Args:
            base_shader: Base shader name ('shadow', 'diffuse', etc.)
            material: Optional Material instance
            extra_defines: Additional shader defines

        Returns:
            Compiled shader program
        """
        # Use default material if none is provided
        mat = material or self.default_material

        # Set up shader defines based on material properties
        defines = {
            'INSTANCED': '1',  # Always use instancing for models
        }

        # Add material properties as defines
        defines.update(mat.to_defines())

        # Add extra defines
        defines.update(extra_defines)

        # Load and return the shader
        return load_shader(base_shader, **defines)

    def load_wavefront(self, filename: str, material: Optional[Material] = None, capacity: int = 100) -> WavefrontModel:
        """
        Load an OBJ model from the assets package and create its VAO and buffers.

        Args:
            filename: Name of the OBJ file in the assets package
            material: Optional Material instance for the model
            capacity: Initial instance capacity

        Returns:
            The loaded WavefrontModel
        """
        # Determine the appropriate shader based on material properties
        mat = material or self.default_material

        files = importlib.resources.files('riverborn')
        with importlib.resources.as_file(files / f'models/{filename}') as model_path:
            mesh = pywavefront.Wavefront(str(model_path), create_materials=True, collect_faces=True)
        model = WavefrontModel(mesh, self.ctx, capacity, material=mat)
        self.models[filename] = model
        return model

    def create_terrain(self, name: str, segments: int, width: float, depth: float,
                       height: float, noise_scale: float,
                       texture: Optional[Union[moderngl.Texture, np.ndarray]] = None,
                       material: Optional[Material] = None) -> TerrainModel:
        """
        Create a terrain model and add it to the scene.

        Args:
            name: Name to identify the terrain model
            segments: Number of grid segments
            width: Width of the terrain
            depth: Depth of the terrain
            height: Height multiplier for the terrain
            noise_scale: Scale of the noise function for terrain generation
            texture: Optional texture for the terrain
            material: Optional Material instance for the terrain

        Returns:
            The created TerrainModel
        """
        # Get appropriate shader for terrain
        mat = material or self.default_material
        terrain_mesh = terrain.make_terrain(segments, width, depth, height, noise_scale)
        model = TerrainModel(terrain_mesh, self.ctx, mat, texture)
        self.models[name] = model
        return model

    def load_terrain(self, name: str, width: float, depth: float,
                       texture: Optional[Union[moderngl.Texture, np.ndarray]] = None,
                       material: Optional[Material] = None) -> TerrainModel:
        """
        Create a terrain model and add it to the scene.

        Args:
            name: Name to identify the terrain model
            segments: Number of grid segments
            width: Width of the terrain
            depth: Depth of the terrain
            height: Height multiplier for the terrain
            noise_scale: Scale of the noise function for terrain generation
            texture: Optional texture for the terrain
            material: Optional Material instance for the terrain

        Returns:
            The created TerrainModel
        """
        files = importlib.resources.files()
        with (files / name).open('rb') as f:
            heights = np.load(f)

        terrain_mesh = terrain.blank_terrain(heights.shape[0] - 1, width, depth)
        terrain_mesh.heights[:] = heights
        terrain.recompute_normals(terrain_mesh)

        # Get appropriate shader for terrain
        mat = material or self.default_material
        model = TerrainModel(terrain_mesh, self.ctx, mat, texture)
        self.models[name] = model
        return model

    def add(self, model: Model) -> Instance:
        """
        Add a new instance of a model to the scene.

        Args:
            model: The model to instance

        Returns:
            The new instance
        """
        inst = Instance(model)
        self.instances.append(inst)
        return inst

    def draw(self, camera: Camera, light: Light) -> None:
        """
        Draw all models with their instances in the scene.

        Args:
            camera: Camera object with view and projection matrices
            light: Light object for lighting calculations
        """
        # Check if we need to use shadows
        if light.shadows:
            # Get or create the shadow system from the light
            shadow_system = light.get_shadow_system()

            # Store it on the scene for debug purposes
            # (needed for rendering the shadow map preview)
            self._shadow_system = shadow_system

            # Render depth pass
            shadow_system.render_depth(self)

        vp = camera.get_proj_matrix() * camera.get_view_matrix()
        planes = extract_frustum_planes(vp)

        for model in self.models.values():
            model._current_planes = planes
            model.draw(camera, light)
            model._current_planes = None

    def render_depth(self, viewproj: glm.mat4) -> None:
        """Render the scene to the shadow map from the light's perspective.

        Args:
            scene: The scene to render
        """
        # Render each model in the scene
        for model_name, model in self.models.items():
            # Skip models that don't cast shadows
            material = model.material

            # Skip rendering this model if it doesn't cast shadows
            if material and not material.cast_shadows:
                continue

            # If instances_dirty is set, update the instance buffer
            model.flush_instances()

            # Render each part of the model
            for part in model.parts:
                # Get the depth shader based on the model's properties

                # Get appropriate depth shader
                depth_shader = model._get_depth_shader(part)

                # Update light space matrix uniform for the depth shader
                try:
                    depth_shader.bind(
                        light_space_matrix=viewproj,
                        **part.depth_uniforms
                    )
                except ValueError as e:
                    raise Exception(
                        f"Failed to bind {part}"
                    ) from e

                # Find the right vertex format based on vbo type
                #
                # We need to ignore the normal data because the shader doesn't
                # use it for depth rendering, so the vertex format must contain
                # "3x4" to ignore it
                vertex_format, *attrs = subformat(part.vbotype, texcoord=material.alpha_test)

                # Extract just the position component for depth pass
                vao_args = [
                    (part.vbo, vertex_format, *attrs),
                    (model.instance_buffer, '16f4/i', 'm_model')
                ]

                # Create the VAO (with or without indices)
                if part.ibo:
                    vao = self.ctx.vertex_array(depth_shader, vao_args, part.ibo)
                else:
                    vao = self.ctx.vertex_array(depth_shader, vao_args)

                # Render this part with instancing
                vao.render(instances=model.instance_count)

                # Clean up the temporary VAO
                vao.release()

    def destroy(self) -> None:
        """
        Clean up all models and instances in the scene.
        """
        for model in self.models.values():
            model.destroy()
        self.models.clear()
        self.instances.clear()


# FIXME: not a very complete mapping
SUPPRESS = {
    '2f4': '2x4',
    '2f': '2x4',
    '3f': '3x4',
    '3f4': '3x4',
    '2x4': '2x4',
    '3x4': '3x4',
}


@cache
def subformat(vbotype: tuple[str, ...], *, texcoord: bool = False, normal: bool = False) -> tuple[str, ...]:
    """Create a subformat string for vertex attributes.

    >>> subformat('2f 3f 3f', texcoord=True, normal=False)
    ('2f 3f 3x4', 'in_texcoord_0', 'in_position')
    """
    format, *attrs = vbotype
    types = format.split(' ')
    newtype = []
    newattrs = []
    for t, a in zip(types, attrs):
        match a:
            case 'in_texcoord_0':
                if texcoord:
                    newattrs.append(a)
                else:
                    t = SUPPRESS[t]
            case 'in_normal':
                if normal:
                    newattrs.append(a)
                else:
                    t = SUPPRESS[t]
            case _:
                newattrs.append(a)
        newtype.append(t)
    return (' '.join(newtype), *newattrs)


# Create a noise texture for terrain
def create_noise_texture(size: int = 256, color=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Generate a texture with Perlin noise."""
    tex_width, tex_height = size, size
    texture_data = np.zeros((tex_height, tex_width, 3), dtype=np.uint8)
    texture_noise_scale = 0.05
    for i in range(tex_height):
        for j in range(tex_width):
            t = noise.pnoise2(
                j * texture_noise_scale,
                i * texture_noise_scale,
                octaves=4,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=1024,
                repeaty=1024,
                base=24,
            )
            c = (t + 1) * 0.5
            texture_data[i, j] = tuple(int(c * comp * 255) for comp in color)
    # Flip vertically to account for texture coordinate differences.
    return np.flipud(texture_data)
