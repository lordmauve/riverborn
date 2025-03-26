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

import importlib.resources
import pywavefront
import moderngl
import numpy as np
import glm
import random
import math
from typing import List, Dict


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
        self.instance_matrices = np.zeros((capacity, 16), dtype='f4')
        self.instances_dirty = False
        self.instance_buffer = ctx.buffer(reserve=capacity * 16 * 4)

        vertex_stride = 8
        for mesh_name, mesh_obj in mesh.meshes.items():
            for material in mesh_obj.materials:
                vertices = np.array(material.vertices, dtype='f4')
                vbo = ctx.buffer(vertices.tobytes())
                if hasattr(material, 'faces') and material.faces:
                    indices = np.array([i for face in material.faces for i in face], dtype='i4')
                    ibo = ctx.buffer(indices.tobytes())
                    vao = ctx.vertex_array(program, [
                        (vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0'),
                        (self.instance_buffer, '4f 4f 4f 4f/i', 'instance_model')
                    ], ibo)
                    self.parts.append({"vbo": vbo, "ibo": ibo, "vao": vao, "indexed": True})
                else:
                    vao = ctx.vertex_array(program, [
                        (vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0'),
                        (self.instance_buffer, '4f 4f 4f 4f/i', 'instance_model')
                    ])
                    self.parts.append({"vbo": vbo, "vao": vao, "indexed": False})

    def add_instance(self, matrix: glm.mat4) -> int:
        """
        Add a new instance matrix to the model, resizing if capacity is exceeded.

        Args:
            matrix: The transformation matrix for the new instance

        Returns:
            Index of the instance in the buffer
        """
        if self.instance_count >= self.instance_capacity:
            self.instance_capacity *= 2
            self.instance_matrices.resize((self.instance_capacity, 16), refcheck=False)
            self.instance_buffer.orphan(size=self.instance_capacity * 16 * 4)
        index = self.instance_count
        self.instance_matrices[index] = np.array(matrix, dtype='f4').flatten()
        self.instance_count += 1
        self.instances_dirty = True
        return index

    def update_instance(self, index: int, matrix: glm.mat4) -> None:
        """
        Update the transformation matrix for an existing instance.

        Args:
            index: Index into the instance matrix buffer
            matrix: New transformation matrix
        """
        self.instance_matrices[index] = np.array(matrix, dtype='f4').flatten()
        self.instances_dirty = True

    def draw(self, m_proj: glm.mat4, m_cam: glm.mat4, sun_dir: glm.vec3) -> None:
        """
        Render the model using instanced rendering.

        Args:
            m_proj: Projection matrix
            m_cam: Camera/view matrix
            sun_dir: Direction of sunlight for lighting calculations
        """
        self.program['m_proj'].write(np.array(m_proj, dtype='f4').tobytes())
        self.program['m_cam'].write(np.array(m_cam, dtype='f4').tobytes())
        self.program['sun_dir'].value = tuple(sun_dir)

        if self.instances_dirty:
            self.instance_buffer.write(self.instance_matrices[:self.instance_count].tobytes())
            self.instances_dirty = False

        for part in self.parts:
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
        self.pos = glm.vec3(0, 0, 0)
        self.rot = glm.quat(1, 0, 0, 0)
        self.scale = glm.vec3(1, 1, 1)
        self.index = model.add_instance(self.matrix)

    @property
    def matrix(self) -> glm.mat4:
        """Compute transformation matrix for this instance."""
        return glm.translate(glm.mat4(1), self.pos) * glm.mat4_cast(self.rot) * glm.scale(glm.mat4(1), self.scale)

    def update(self) -> None:
        """
        Recalculate and upload the transformation matrix to the model's buffer.
        """

        self.model.update_instance(self.index, self.matrix)

class Scene:
    """
    A simple container for models and their instances.

    Supports loading models from the assets package and drawing by model name.
    """
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
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
        with importlib.resources.path('assets', filename) as model_path:
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

    def draw(self, model_filename: str, m_proj: glm.mat4, m_cam: glm.mat4, sun_dir: glm.vec3) -> None:
        """
        Draw all instances of a given model.

        Args:
            model_filename: Filename of the model as loaded via `load()`
            m_proj: Projection matrix
            m_cam: Camera/view matrix
            sun_dir: Directional light vector
        """
        if model_filename in self.models:
            self.models[model_filename].draw(m_proj, m_cam, sun_dir)

    def destroy(self) -> None:
        """
        Clean up all models and instances in the scene.
        """
        for model in self.models.values():
            model.destroy()
        self.models.clear()
        self.instances.clear()
