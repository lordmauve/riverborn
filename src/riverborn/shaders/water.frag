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

// Smooth normal calculation function
vec3 calculateSmoothNormal(sampler2D heightMap, vec2 uv, float ripple_scale, float smoothness) {
    // Get texture dimensions for proper offset calculation
    vec2 texelSize = 1.0 / textureSize(heightMap, 0);

    // Sample heights at multiple points for a smoother gradient
    float h_c = texture(heightMap, uv).r;
    float h_r = texture(heightMap, uv + vec2(texelSize.x, 0.0)).r;
    float h_l = texture(heightMap, uv + vec2(-texelSize.x, 0.0)).r;
    float h_u = texture(heightMap, uv + vec2(0.0, texelSize.y)).r;
    float h_d = texture(heightMap, uv + vec2(0.0, -texelSize.y)).r;

    // Also sample at diagonals for better smoothing
    float h_ur = texture(heightMap, uv + texelSize).r;
    float h_ul = texture(heightMap, uv + vec2(-texelSize.x, texelSize.y)).r;
    float h_dr = texture(heightMap, uv + vec2(texelSize.x, -texelSize.y)).r;
    float h_dl = texture(heightMap, uv + vec2(-texelSize.x, -texelSize.y)).r;

    // Calculate gradients using Sobel filter for better smoothing
    float dx = (h_ur + 2.0 * h_r + h_dr) - (h_ul + 2.0 * h_l + h_dl);
    float dy = (h_ul + 2.0 * h_u + h_ur) - (h_dl + 2.0 * h_d + h_dr);

    // Apply smoothness factor to control the effect
    dx *= ripple_scale * smoothness;
    dy *= ripple_scale * smoothness;

    // Construct and return the normal
    return normalize(vec3(-dx, 1.0, -dy));
}

void main() {
    // --- Compute perturbed normal from the height map.
    float h = texture(height_map, v_uv).r;

    // Original method (commented out)
    // float dFdx_h = dFdx(h);
    // float dFdy_h = dFdy(h);
    // float ripple_scale = 1;
    // vec3 perturbed_normal = normalize(vec3(-dFdx_h * ripple_scale, 1.0, -dFdy_h * ripple_scale));

    // Use the new smooth normal calculation
    float ripple_scale = 0.6;
    float smoothness = 0.1; // Adjust this value to control smoothing strength (lower = smoother)
    vec3 perturbed_normal = calculateSmoothNormal(height_map, v_uv, ripple_scale, smoothness);

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
    if (depth_diff < 0.0) {
        // If the water is above the scene, we want it to be opaque.
        discard;
    }
    // When the water is shallow (small depth difference) we want more transparency.
    float shallow = clamp(depth_diff / water_opaque_depth, 0.0, 1.0);

    // Mix: reflection atop base water colour.
    vec4 diffuse = vec4(base_water, shallow);
    f_color = mix(diffuse, refl_color, fresnel);
}
