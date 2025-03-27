#version 330
in vec3 in_position;

#ifdef INSTANCED
in mat4 m_model;
#else
uniform mat4 m_model;
#endif
uniform mat4 light_space_matrix;

void main() {
    gl_Position = light_space_matrix * m_model * vec4(in_position, 1.0);
}
