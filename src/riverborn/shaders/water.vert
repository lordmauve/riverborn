#version 330
in vec3 in_position;
in vec2 in_uv;

uniform mat4 m_model;
uniform mat4 m_proj;
uniform mat4 m_view;

out vec2 v_uv;
out vec3 v_world;


void main() {
    v_uv = in_uv;
    vec4 world = m_model * vec4(in_position, 1.0);
    v_world = world.xyz;
    gl_Position = m_proj * m_view * world;
}
