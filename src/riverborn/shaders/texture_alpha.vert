#version 330

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

in mat4 m_model;
//uniform mat4 m_model;

uniform mat4 m_proj;
uniform mat4 m_view;

out vec3 normal;
out vec2 uv;
out vec3 pos;

void main() {
    mat4 mv = m_view * transpose(m_model);
    mat3 normal_matrix = inverse(mat3(mv));
    normal = normalize(normal_matrix * in_normal);

    gl_Position = m_proj * mv * vec4(in_position, 1.0);
    pos = gl_Position.xyz;

    uv = in_texcoord_0;
}
