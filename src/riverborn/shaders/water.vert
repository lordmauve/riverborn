#version 330
in vec3 in_position;
in vec2 in_uv;
uniform mat4 mvp;
uniform mat4 model;
out vec2 v_uv;
out vec3 v_world;
void main() {
    v_uv = in_uv;
    vec4 world = model * vec4(in_position, 1.0);
    v_world = world.xyz;
    gl_Position = mvp * vec4(in_position, 1.0);
}
