import numpy as np
from pyrr import Matrix44, Vector3


class Camera:
    def __init__(self, eye, target, up, fov=70.0, aspect=1.0, near=0.1, far=1000.0):
        self.eye = Vector3(eye, dtype='f4')
        self.target = Vector3(target, dtype='f4')
        self.up = Vector3(up)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        self._dirty = True
        self._view_matrix = None
        self._proj_matrix = None

    def update_matrices(self):
        self._view_matrix = Matrix44.look_at(self.eye, self.target, self.up, dtype='f4')
        self._proj_matrix = Matrix44.perspective_projection(
            self.fov, self.aspect, self.near, self.far,
            dtype='f4',
        )
        self._dirty = False

    def get_view_matrix(self):
        if self._dirty or self._view_matrix is None:
            self.update_matrices()
        return self._view_matrix

    def get_proj_matrix(self):
        if self._dirty or self._proj_matrix is None:
            self.update_matrices()
        return self._proj_matrix

    def set_aspect(self, aspect):
        if aspect != self.aspect:
            self.aspect = aspect
            self._dirty = True

    def bind(self, prog, model, normal_matrix: str | None = None, mvp_uniform: str | None = 'mvp', pos: str | None = None):
        """
        Bind the computed MVP and normal matrix to the provided shader program.
        :param prog: The shader program (dictionary-like uniform access).
        :param model: The model transformation matrix (Matrix44).
        :param normal_matrix: Name of the normal matrix uniform.
        :param mvp_uniform: Name of the mvp matrix uniform.
        """
        view = self.get_view_matrix()
        proj = self.get_proj_matrix()
        mvp = proj * view * model.astype('f4')

        prog[mvp_uniform].write(mvp)

        if normal_matrix:
            # Compute the normal matrix (inverse-transpose of the model's upper 3x3).
            nm = np.linalg.inv(model[:3, :3]).T
            prog[normal_matrix].write(nm.astype("f4").tobytes())
        if pos:
            prog[pos].value = self.eye
