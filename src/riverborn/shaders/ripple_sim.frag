#version 330
uniform sampler2D curr_tex;
uniform vec2 texel;

in vec2 v_texcoord;
out vec4 fragColor;

const float dt = 1.0;

// higher tension means faster ripples
const float tension = 0.05;
const float damping_factor = 0.99;

void main() {
    float hL = texture(curr_tex, v_texcoord - vec2(texel.x, 0.0)).r;
    float hR = texture(curr_tex, v_texcoord + vec2(texel.x, 0.0)).r;
    float hD = texture(curr_tex, v_texcoord - vec2(0.0, texel.y)).r;
    float hU = texture(curr_tex, v_texcoord + vec2(0.0, texel.y)).r;
    vec2 here = texture(curr_tex, v_texcoord).rg;

    float laplacian = hL + hR + hD + hU - 4.0 * here.r;
    float velocity = here.g + tension * laplacian * dt;
    velocity *= damping_factor;
    float new_height = here.r + velocity * dt;

    fragColor = vec4(new_height, velocity, 0.0, 1.0);
}
