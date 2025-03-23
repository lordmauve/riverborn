#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

uniform mat4 mvp;
uniform mat3 normal_matrix;

out vec3 frag_normal;
out vec2 frag_uv;

void main() {
    frag_uv = in_uv;
    frag_normal = normalize(normal_matrix * in_normal);
    gl_Position = mvp * vec4(in_position, 1.0);
}
