from pyglm import glm


class Camera:
    def __init__(self, eye, target, up = glm.vec3(0, 1, 0), fov=70.0, aspect=1.0, near=0.1, far=1000.0):
        self._eye = glm.vec3(eye)
        self._target = glm.vec3(target)
        self._up = glm.vec3(up)
        self._fov = fov
        self._aspect = aspect
        self._near = near
        self._far = far

        self._view_matrix = None
        self._proj_matrix = None

    @property
    def eye(self):
        return self._eye

    @eye.setter
    def eye(self, value):
        self._eye = glm.vec3(value)
        self._view_matrix = None

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = glm.vec3(value)
        self._view_matrix = None

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, value):
        self._up = glm.vec3(value)
        self._view_matrix = None

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = value
        self._proj_matrix = None

    @property
    def aspect(self):
        return self._aspect

    @aspect.setter
    def aspect(self, value):
        self._aspect = value
        self._proj_matrix = None

    @property
    def near(self):
        return self._near

    @near.setter
    def near(self, value):
        self._near = value
        self._proj_matrix = None

    @property
    def far(self):
        return self._far

    @far.setter
    def far(self, value):
        self._far = value
        self._proj_matrix = None

    def update_matrices(self):
        self._view_matrix = glm.lookAt(self._eye, self._target, self._up)
        self._proj_matrix = glm.perspective(
            self._fov, self._aspect, self._near, self._far,
        )

    def look_at(self, target):
        self.target = glm.vec3(target)

    def get_view_matrix(self):
        if self._view_matrix is None:
            self.update_matrices()
        return self._view_matrix

    def get_proj_matrix(self):
        if self._proj_matrix is None:
            self.update_matrices()
        return self._proj_matrix

    def set_aspect(self, aspect):
        if aspect != self._aspect:
            self.aspect = aspect

    def bind(self, prog, view_uniform: str = 'm_view', proj_uniform: str = 'm_proj', pos_uniform: str | None = None):
        view = self.get_view_matrix()
        proj = self.get_proj_matrix()

        prog[view_uniform].write(view)
        prog[proj_uniform].write(proj)

        if pos_uniform:
            prog[pos_uniform].value = self._eye
