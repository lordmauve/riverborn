// Shadow mapping parameters
uniform sampler2D shadow_map;
uniform mat4 light_space_matrix;


float calculate_shadow(vec4 light_space_pos, float bias) {
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;

    // Check if fragment is outside the shadow map
    if(proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
       proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
       proj_coords.z < 0.0 || proj_coords.z > 1.0) {
        return 0.0;
    }

    float closest_depth = texture(shadow_map, proj_coords.xy).r;
    float current_depth = proj_coords.z;

    // Apply bias to avoid shadow acne
    float shadow = current_depth - bias > closest_depth ? 1.0 : 0.0;
    return shadow;
}

// PCF (Percentage-Closer Filtering) version for smoother shadows
float calculate_shadow_pcf(vec4 light_space_pos, float bias, float pcf_radius) {
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;

    // Check if fragment is outside the shadow map
    if(proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
       proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
       proj_coords.z < 0.0 || proj_coords.z > 1.0) {
        return 0.0;
    }

    float current_depth = proj_coords.z;
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(shadow_map, 0);

    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcf_depth = texture(shadow_map, proj_coords.xy + vec2(x, y) * texel_size * pcf_radius).r;
            shadow += current_depth - bias > pcf_depth ? 1.0 : 0.0;
        }
    }

    shadow /= 9.0;
    return shadow;
}
