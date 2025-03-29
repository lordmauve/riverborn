#version 330
// Water plane fragment shader.
in vec2 v_uv;
in vec3 v_world;
uniform sampler2D height_map;   // Height map for ripples.
uniform samplerCube env_cube;     // Environment cube map.
uniform sampler2D depth_tex;      // Depth texture from water-bottom pass.
uniform vec3 camera_pos;
uniform vec2 resolution;
uniform float near;
uniform float far;
uniform vec3 base_water;
uniform float water_opaque_depth;
out vec4 f_color;

// Function to linearize a non-linear depth value.
float linearizeDepth(float depth) {
    return (2.0 * near * far) / (far + near - depth * (far - near));
}

void main() {
    // --- Compute perturbed normal from the height map.
    float h = texture(height_map, v_uv).r;
    // Derivatives of the height value (for a simple normal perturbation).
    float dFdx_h = dFdx(h);
    float dFdy_h = dFdy(h);
    float ripple_scale = 1.0;
    vec3 perturbed_normal = normalize(vec3(-dFdx_h * ripple_scale, 1.0, -dFdy_h * ripple_scale));

    // --- Fresnel term.
    vec3 view_dir = normalize(camera_pos - v_world);
    float fresnel = pow(1.0 - max(dot(perturbed_normal, view_dir), 0.0), 3.0);

    // --- Reflection from the environment.
    vec3 refl_dir = reflect(-view_dir, perturbed_normal);
    vec4 refl_color = vec4(texture(env_cube, refl_dir).rgb, 1.0);

    // --- Depth-based transparency.
    // Compute screen-space coordinates.
    vec2 screen_uv = gl_FragCoord.xy / resolution;
    // Sample the water-bottom depth (non-linear depth).
    float scene_depth = texture(depth_tex, screen_uv).r;
    // Linearize the depths.
    float scene_lin = linearizeDepth(scene_depth);
    float water_lin = linearizeDepth(gl_FragCoord.z);
    float depth_diff = scene_lin - water_lin;
    // When the water is shallow (small depth difference) we want more transparency.
    float shallow = clamp(depth_diff / water_opaque_depth, 0.0, 1.0);

    // Mix: reflection atop base water colour.
    vec4 diffuse = vec4(base_water, shallow);
    f_color = mix(diffuse, refl_color, fresnel);
}
