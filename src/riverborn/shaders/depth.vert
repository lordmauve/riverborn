#version 330
in vec3 in_position;

#ifdef INSTANCED
in mat4 m_model;
#else
uniform mat4 m_model;
#endif
uniform mat4 light_space_matrix;

#ifdef ALPHA_TEST
in vec2 in_texcoord_0;
out vec2 frag_texcoord_0;
#endif

void main() {
#ifdef ALPHA_TEST
    frag_texcoord_0 = in_texcoord_0;
#endif
    vec4 world_pos = m_model * vec4(in_position, 1.0);
    gl_Position = light_space_matrix * world_pos;
}
