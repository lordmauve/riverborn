#version 330
uniform sampler2D height_tex;
uniform vec2 texel;
uniform vec3 light_dir;  // direction of the light (e.g. from top-left)
in vec2 v_texcoord;
out vec4 fragColor;
void main() {
    float hL = texture(height_tex, v_texcoord - vec2(texel.x, 0.0)).r;
    float hR = texture(height_tex, v_texcoord + vec2(texel.x, 0.0)).r;
    float hD = texture(height_tex, v_texcoord - vec2(0.0, texel.y)).r;
    float hU = texture(height_tex, v_texcoord + vec2(0.0, texel.y)).r;
    // Compute the approximate normal: the z component is set to 1.0 for scale.
    vec3 normal = normalize(vec3(hL - hR, hD - hU, 1.0));
    float diff = max(dot(normal, normalize(light_dir)), 0.0);
    // Base water color modulated by diffuse lighting.
    vec3 baseColor = vec3(0.0, 0.5, 1.0);
    fragColor = vec4(baseColor * diff, 1.0);
}
