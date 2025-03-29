#version 330
#include "shadow_common.glsl"

in vec3 frag_normal;
in vec3 frag_pos;
in vec4 frag_pos_light_space;

#ifdef TEXTURE
in vec2 frag_uv;
uniform sampler2D diffuse_tex;
#endif

uniform vec3 light_dir;
uniform vec3 light_color;
uniform vec3 ambient_color;
uniform vec3 camera_pos;
const vec3 transmissivity = vec3(0.2, 0.2, 0.2);  // How much light passes through the surface

uniform float shadow_bias = 0.001;
uniform float pcf_radius = 1.0;
uniform bool use_pcf = true;

out vec4 fragColor;

void main() {
#ifdef TEXTURE
    // Sample texture
    vec4 tex_color = texture(diffuse_tex, frag_uv);
    if (tex_color.a < 0.3) {
        discard;
    }
#else
    vec4 tex_color = vec4(1.0);
#endif

    // Get the normal and check if we're viewing the back face
    vec3 normal = normalize(frag_normal);
    vec3 view_dir = normalize(camera_pos - frag_pos);

    // Check if we're viewing the back face
    bool is_back_face = dot(view_dir, normal) < 0.0;

    // Flip the normal for back faces
    if (is_back_face) {
        normal = -normal;
    }

    vec3 light = normalize(light_dir);

    // Ambient
    vec3 ambient = ambient_color;

    // Diffuse
    float diff = max(dot(normal, light), 0.0);
    vec3 diffuse = diff * light_color;

    // For back faces, add transmitted light based on transmissivity
    // This simulates light passing through the surface from the front
    if (is_back_face) {
        // Calculate light hitting the front face (opposite normal)
        float front_diff = max(dot(-normal, light), 0.0);
        // Add transmitted light based on transmissivity parameter
        diffuse += front_diff * light_color * transmissivity;
    }

    // Specular (optional)
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
