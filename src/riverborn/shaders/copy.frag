#version 330 core

// Input texture coordinates from the vertex shader
in vec2 frag_texcoord;

// The texture to copy
uniform sampler2D input_texture;

// Output color
out vec4 fragColor;

void main() {
    // Sample the input texture and output the color
    fragColor = texture(input_texture, frag_texcoord);
}
