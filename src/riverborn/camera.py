from pyglm import glm


class Camera:
    def __init__(self, eye, target, up = glm.vec3(0, 1, 0), fov=70.0, aspect=1.0, near=0.1, far=1000.0):
        self.eye = glm.vec3(eye)
        self.target = glm.vec3(target)
        self.up = glm.vec3(up)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        self._dirty = True
        self._view_matrix = None
        self._proj_matrix = None

    def update_matrices(self):
        self._view_matrix = glm.lookAt(self.eye, self.target, self.up)
        self._proj_matrix = glm.perspective(
            self.fov, self.aspect, self.near, self.far,
        )
        self._dirty = False

    def look_at(self, target):
        self.target = glm.vec3(target)
        self._dirty = True

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

    def bind(self, prog, view_uniform: str = 'm_view', proj_uniform: str = 'm_proj', pos_uniform: str | None = None):

        view = self.get_view_matrix()
        proj = self.get_proj_matrix()

        prog[view_uniform].write(view)
        prog[proj_uniform].write(proj)

        if pos_uniform:
            prog[pos_uniform].value = self.eye
