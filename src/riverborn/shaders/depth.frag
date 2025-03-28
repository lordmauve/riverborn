#version 330
// Empty fragment shader for depth pass

#ifdef ALPHA_TEST
const float alpha_test = 0.3;
uniform sampler2D diffuse_tex;
in vec2 frag_texcoord_0;
#endif

void main() {
#ifdef ALPHA_TEST
    vec4 tex_color = texture(diffuse_tex, frag_texcoord_0);
    if (tex_color.a < alpha_test) {
        discard;
    }
#endif
}
