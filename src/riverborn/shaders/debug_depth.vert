#version 330
in vec3 in_position;
in vec2 in_texcoord_0;
out vec2 frag_uv;
void main() {
    frag_uv = in_texcoord_0;
    gl_Position = vec4(in_position, 1.0);
}
