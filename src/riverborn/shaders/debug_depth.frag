#version 330
in vec2 frag_uv;
uniform sampler2D shadow_map;
uniform float near_plane = 1.0;
uniform float far_plane = 100.0;
out vec4 fragColor;

// Function to linearize depth for better visualization
float linearize_depth(float depth) {
    float z = depth * 2.0 - 1.0; // Convert from [0,1] to [-1,1]
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
}

void main() {
    // Sample the shadow map (depth texture)
    float depth_value = texture(shadow_map, frag_uv).r;

    // Check if the depth value is valid
    if (depth_value == 1.0) {
        // Red for far plane (helps diagnose if the entire texture is at the far plane)
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else if (depth_value == 0.0) {
        // Blue for near plane or uninitiated values
        fragColor = vec4(0.0, 0.0, 1.0, 1.0);
    } else {
        // Use a simple grayscale visualization with enhanced contrast
        float adjusted = pow(depth_value, 0.5); // Gamma correction for better visibility
        fragColor = vec4(vec3(adjusted), 1.0);

        // Uncomment to debug with linearized depth if needed
        // float linear_depth = linearize_depth(depth_value);
        // linear_depth = linear_depth / far_plane; // Scale to 0-1 range
        // fragColor = vec4(vec3(linear_depth), 1.0);
    }

    // Debug checkerboard background to ensure shader is working
    vec2 checker = floor(frag_uv * 8.0);
    float checkerValue = mod(checker.x + checker.y, 2.0);

    // Add checkerboard pattern in the background
    fragColor = mix(
        vec4(vec3(0.1) + 0.05 * checkerValue, 1.0),
        fragColor,
        0.8
    );
}
