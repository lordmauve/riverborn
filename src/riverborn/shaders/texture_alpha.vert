#version 330

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

in mat4 m_model;

uniform mat4 m_proj;
uniform mat4 m_view;

out vec3 normal;
out vec2 uv;
out vec3 pos;

void main() {
    // Compute the model-view matrix
    mat4 mv = m_view * transpose(m_model);

    // Compute the normal matrix (inverse transpose of the upper-left 3x3 of mv)
    mat3 normal_matrix = transpose(inverse(mat3(mv)));

    // Transform the normal using the normal matrix
    normal = normalize(normal_matrix * in_normal);

    // Transform the position to view space
    vec4 view_pos = mv * vec4(in_position, 1.0);
    pos = view_pos.xyz;

    // Pass the texture coordinates
    uv = in_texcoord_0;

    // Compute the final position in clip space
    gl_Position = m_proj * view_pos;
}
