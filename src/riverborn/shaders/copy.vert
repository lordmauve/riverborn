#version 330 core

// Vertex attributes for a fullscreen quad
in vec2 in_position;
in vec2 in_texcoord_0;

// Pass texture coordinates to the fragment shader
out vec2 frag_texcoord;

void main() {
    frag_texcoord = in_texcoord_0;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
