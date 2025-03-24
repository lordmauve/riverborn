#version 330
in vec2 in_position;
out vec2 v_texcoord;
void main() {
    // in_position is in the range [-1,1] for a full-screen quad
    v_texcoord = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
