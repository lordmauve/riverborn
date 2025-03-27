"""
Scene system for instanced rendering using ModernGL and PyWavefront.

This module defines a minimal scene management system supporting instanced rendering
of OBJ models using ModernGL. Each model holds a buffer of instance transformation
matrices. These matrices are used to render multiple instances of the same mesh
with different transforms using a single draw call.

The system supports:
- Loading OBJ + MTL files with PyWavefront
- Creating a persistent instance buffer per model
- Resizing the buffer dynamically with orphan()
- Dirty flagging to only update GPU buffers when needed
- Directional (sunlight) lighting with diffuse and specular shading
"""

from contextlib import suppress
import importlib.resources
from pathlib import Path
import imageio
import pywavefront
import moderngl
import moderngl_window as mglw
import numpy as np
from pyglm import glm
import random
import math
import weakref
from typing import List, Dict

from riverborn.camera import Camera
from riverborn.shader import load_shader


SUN_DIR = glm.normalize(glm.vec3(1, 1, 1))



class Model:
    """
    A 3D model loaded from an OBJ file with support for instanced rendering.

    Attributes:
        ctx: ModernGL context
        program: Shader program used for this model
        parts: List of model parts (per material) each with its own VAO
        instance_matrices: CPU-side array of transformation matrices
        instance_buffer: GPU buffer for instance matrices
        instance_capacity: Current allocated capacity (resized as needed)
        instance_count: Number of active instances
        instances_dirty: Flag indicating whether GPU buffer needs update
    """

    textures = {}

    def __init__(self, mesh: pywavefront.Wavefront, ctx: moderngl.Context, program: moderngl.Program, capacity: int = 100) -> None:
        """
        Initialize the model by creating VAOs for each mesh part and reserving
        space for per-instance data.

        Args:
            mesh: PyWavefront mesh object
            ctx: ModernGL context
            program: Shader program to use
            capacity: Initial number of instances to support
        """
        self.ctx = ctx
        self.program = program
        self.parts = []
        self.instance_capacity = capacity
        self.instance_count = 0
        self.instance_matrices = np.zeros((capacity, 4, 4), dtype='f4')
        self.instances_dirty = False
        self.instance_buffer = ctx.buffer(reserve=capacity * 16 * 4)
        self.instance_refs = weakref.WeakValueDictionary()

        vertex_stride = 8
        for mesh_name, mesh_obj in mesh.meshes.items():
            for material in mesh_obj.materials:
                assert material.vertex_format == 'T2F_N3F_V3F', \
                    f"Unsupported vertex format: {material.vertex_format}"
                vertices = np.array(material.vertices, dtype='f4')
                vbo = ctx.buffer(vertices.tobytes())
                if hasattr(material, 'texture'):
                    path = Path(material.texture.path).name
                    texture = {'texture': self.load_texture(path)}
                else:
                    texture = {}

                if hasattr(material, 'faces') and material.faces:
                    indices = np.array([i for face in material.faces for i in face], dtype='i4')
                    ibo = ctx.buffer(indices.tobytes())
                    vao = ctx.vertex_array(program, [
                        (vbo, '2f 3f 3f', 'in_texcoord_0', 'in_normal', 'in_position'),
                        (self.instance_buffer, '16f4/i', 'm_model')
                    ], ibo)
                    self.parts.append({
                        "vbo": vbo, "ibo": ibo, "vao": vao, "indexed": True,
                        **texture
                    })
                else:
                    vao = ctx.vertex_array(program, [
                        (vbo, '2f 3f 3f', 'in_texcoord_0', 'in_normal', 'in_position'),
                        (self.instance_buffer, '16f4/i', 'm_model')
                    ])
                    self.parts.append({
                        "vbo": vbo,
                        "vao": vao,
                        "indexed": False,
                        **texture
                    })

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

    def add_instance(self, matrix: glm.mat4, instance=None) -> int:
        """
        Add a new instance matrix to the model, resizing if capacity is exceeded.

        Args:
            matrix: The transformation matrix for the new instance
            instance: The Instance object associated with this matrix (optional)

        Returns:
            Index of the instance in the buffer
        """
        if self.instance_count >= self.instance_capacity:
            self.instance_capacity *= 2
            self.instance_matrices.resize((self.instance_capacity, 4, 4), refcheck=False)
            self.instance_buffer.orphan(size=self.instance_capacity * 16 * 4)
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

    def draw(self, camera: Camera, sun_dir: glm.vec3) -> None:
        """
        Render the model using instanced rendering.

        Args:
            m_proj: Projection matrix
            m_cam: Camera/view matrix
            sun_dir: Direction of sunlight for lighting calculations
        """
        camera.bind(self.program)

        with suppress(KeyError):
            self.program['sun_dir'].value = tuple(sun_dir)

        if True or self.instances_dirty:
            self.instance_buffer.write(self.instance_matrices[:self.instance_count])
            self.instances_dirty = False

        for part in self.parts:
            if texture := part.get('texture'):
                texture.use(location=0)
            part['vao'].render(instances=self.instance_count)

    def destroy(self) -> None:
        """
        Clean up GPU resources used by this model.
        """
        for part in self.parts:
            part['vbo'].release()
            if part.get('indexed'):
                part['ibo'].release()
            part['vao'].release()
        self.instance_buffer.release()
        self.parts.clear()


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
        if self._deleted:
            return

        last_index = self.model.instance_count - 1
        if self.index != last_index:
            # Get the instance that uses the last slot
            last_instance_ref = self.model.instance_refs.get(last_index)
            last_instance = last_instance_ref() if last_instance_ref else None

            if last_instance:
                # Update its index to point to the slot we're removing
                last_instance.index = self.index

                # Update the instance reference mapping
                self.model.instance_refs[self.index] = self.model.instance_refs[last_index]

            # Copy the last matrix to this slot
            self.model.instance_matrices[self.index] = self.model.instance_matrices[last_index]
            self.model.instances_dirty = True

        # Remove the reference to the last slot
        if last_index in self.model.instance_refs:
            del self.model.instance_refs[last_index]

        self.model.instance_count -= 1
        self._deleted = True


class Scene:
    """
    A simple container for models and their instances.

    Supports loading models from the assets package and drawing by model name.
    """
    def __init__(self) -> None:
        self.ctx = mglw.ctx()
        self.models: Dict[str, Model] = {}
        self.instances: List[Instance] = []

    def load(self, filename: str, program: moderngl.Program, capacity: int = 100) -> Model:
        """
        Load an OBJ model from the assets package and create its VAO and buffers.

        Args:
            filename: Name of the OBJ file in the assets package
            program: Shader program to use
            capacity: Initial instance capacity

        Returns:
            The loaded Model
        """
        files = importlib.resources.files()
        with importlib.resources.as_file(files / f'models/{filename}') as model_path:
            mesh = pywavefront.Wavefront(str(model_path), create_materials=True, collect_faces=True)
        model = Model(mesh, self.ctx, program, capacity)
        self.models[filename] = model
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

    def draw(self, camera: Camera, sun_dir: glm.vec3) -> None:
        """
        Draw all instances of a given model.

        Args:
            model_filename: Filename of the model as loaded via `load()`
            m_proj: Projection matrix
            m_cam: Camera/view matrix
            sun_dir: Directional light vector
        """
        for model in self.models.values():
            model.draw(camera, sun_dir)

    def destroy(self) -> None:
        """
        Clean up all models and instances in the scene.
        """
        for model in self.models.values():
            model.destroy()
        self.models.clear()
        self.instances.clear()


SUN_DIR = glm.normalize(glm.vec3(1, 1, 1))


class SceneDemo(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Instanced rendering demo"
    window_size = (1920, 1080)
    aspect_ratio = None  # Let the window determine the aspect ratio.
    resizable = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create a shader program for the model.
        prog: moderngl.Program = load_shader('texture_alpha')

        self.scene = Scene()
        plant: Model = self.scene.load('fern.obj', prog, capacity=100)

        self.camera = Camera(
            eye=[0.0, 20.0, 50.0],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 1.0, 0.0],
            fov=70.0,
            aspect=self.wnd.aspect_ratio,
            near=0.1,
            far=1000.0,
        )
        self.camera.set_aspect(self.wnd.aspect_ratio)
        self.camera.look_at(glm.vec3(0, 0, 0))

        for _ in range(200):
            inst: Instance = self.scene.add(plant)
            inst.pos = (random.uniform(-100, 100), 1, random.uniform(-100, 100))
            inst.rotate(random.uniform(0, 2 * math.pi), glm.vec3(0, 1, 0))
            inst.scale = glm.vec3(0.1)
            inst.update()

    def on_render(self, time, frame_time):
        self.ctx.clear(0.5, 0.55, 0.7, 1.0)
        # Draw the plant model with all its instances.
        self.scene.draw(self.camera, SUN_DIR)

    def on_close(self):
        self.scene.destroy()


    def on_key_event(self, key, action, modifiers):

        op = 'press' if action == self.wnd.keys.ACTION_PRESS else 'release'
        keys = self.wnd.keys
        match op, key, modifiers.shift:
            case ('press', keys.ESCAPE, _):
                sys.exit()

            case ('press', keys.F12, False):
                from .screenshot import screenshot
                screenshot()


def main():
    mglw.run_window_config(SceneDemo)
