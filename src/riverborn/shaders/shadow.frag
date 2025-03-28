#version 330
#include "shadow_common.glsl"

in vec3 frag_normal;
in vec2 frag_uv;
in vec3 frag_pos;
in vec4 frag_pos_light_space;

uniform sampler2D diffuse_tex;
uniform vec3 light_dir;
uniform vec3 light_color;
uniform vec3 ambient_color;
uniform vec3 camera_pos;

uniform float shadow_bias = 0.005;
uniform float pcf_radius = 1.0;
uniform bool use_pcf = true;

out vec4 fragColor;

void main() {
    // Sample texture
    vec4 tex_color = texture(diffuse_tex, frag_uv);
    if (tex_color.a < 0.3) {
        discard;
    }

    vec3 normal = normalize(frag_normal);
    vec3 light = normalize(light_dir);

    // Ambient
    vec3 ambient = ambient_color;

    // Diffuse
    float diff = max(dot(normal, light), 0.0);
    vec3 diffuse = diff * light_color;

    // Specular (optional)
    vec3 view_dir = normalize(camera_pos - frag_pos);
    vec3 reflect_dir = reflect(-light, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    vec3 specular = 0.2 * spec * light_color;

    // Calculate shadow
    float shadow;
    if (use_pcf) {
        shadow = calculate_shadow_pcf(frag_pos_light_space, shadow_bias, pcf_radius);
    } else {
        shadow = calculate_shadow(frag_pos_light_space, shadow_bias);
    }

    // Combine lighting with shadow
    vec3 lighting = ambient + (1.0 - shadow) * diffuse;


    // Final color
    fragColor = vec4(lighting * tex_color.rgb + specular, 1.0);
}
