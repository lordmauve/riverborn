#version 330 core

in vec2 frag_uv;
out vec4 fragColor;

uniform sampler2D depth_texture;
uniform vec2 resolution;
uniform float near;
uniform float far;

const float radius = 0.5;
const int sample_count = 16;

vec3 get_view_position(float depth, vec2 uv) {
    float z = depth * 2.0 - 1.0;
    float x = (uv.x * 2.0 - 1.0) * (resolution.x / resolution.y);
    float y = uv.y * 2.0 - 1.0;
    return vec3(x, y, z);
}

float linearize_depth(float depth) {
    return (2.0 * near * far) / (far + near - depth * (far - near));
}

float sample_ao(vec2 uv, vec3 view_pos, vec3 normal) {
    float ao = 0.0;
    for (int i = 0; i < sample_count; ++i) {
        vec2 offset = vec2(
            cos(float(i) * 6.28318530718 / float(sample_count)),
            sin(float(i) * 6.28318530718 / float(sample_count))
        ) * radius / resolution;
        float sample_depth = texture(depth_texture, uv + offset).r;
        vec3 sample_pos = get_view_position(sample_depth, uv + offset);
        vec3 delta = sample_pos - view_pos;
        float dist = length(delta);
        float NdotD = max(dot(normal, delta / dist), 0.0);
        ao += (1.0 - NdotD) * smoothstep(0.0, radius, dist);
    }
    return ao / float(sample_count);
}

void main() {
    float depth = texture(depth_texture, frag_uv).r;
    vec3 view_pos = get_view_position(depth, frag_uv);
    vec3 normal = normalize(cross(dFdx(view_pos), dFdy(view_pos)));
    float ao = sample_ao(frag_uv, view_pos, normal);
    fragColor = vec4(vec3(ao), 1.0);
}
