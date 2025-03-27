#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

#ifdef INSTANCED
in mat4 m_model;
#else
uniform mat4 m_model;
#endif

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 light_space_matrix;

out vec3 frag_normal;
out vec2 frag_uv;
out vec3 frag_pos;
out vec4 frag_pos_light_space;

void main() {
    // Transform to world space
    mat4 model = transpose(m_model);

    vec4 world_pos = model * vec4(in_position, 1.0);
    frag_pos = world_pos.xyz;

    // Calculate normal in world space
    mat3 normal_matrix = transpose(inverse(mat3(model)));
    frag_normal = normalize(normal_matrix * in_normal);

    // Pass texture coordinates
    frag_uv = in_texcoord_0;

    // Calculate position in light space for shadow mapping
    frag_pos_light_space = light_space_matrix * world_pos;

    // Calculate clip space position
    gl_Position = m_proj * m_view * world_pos;
}
