#version 330
// Empty fragment shader for depth pass

#ifdef ALPHA_TEST
uniform float alpha_test;
uniform sampler2D texture_0;
in vec2 frag_texcoord_0;
#endif

void main() {
#ifdef ALPHA_TEST
    vec4 tex_color = texture(texture_0, frag_texcoord_0);
    if (tex_color.a < alpha_test) {
        discard;
    }
#endif
}
