#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

uniform mat4 m_proj;
uniform mat4 m_view;

#ifdef INSTANCED
in mat4 m_model;
#else
uniform mat4 m_model;
#endif

out vec3 frag_normal;
out vec2 frag_uv;

void main() {
    frag_uv = in_texcoord_0;
    mat4 mv = m_view * m_model;
    mat3 normal_matrix = inverse(mat3(m_model));
    frag_normal = normalize(normal_matrix * in_normal);
    gl_Position = m_proj * mv * vec4(in_position, 1.0);
}
