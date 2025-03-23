#version 330
in vec3 frag_normal;
in vec2 frag_uv;

uniform sampler2D Texture;
uniform vec3 light_dir;
uniform vec3 light_color;
uniform vec3 ambient_color;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(frag_normal);
    vec3 light = normalize(light_dir);
    float diffuse = max(dot(normal, light), 0.0);
    vec3 tex_color = texture(Texture, frag_uv).rgb;
    vec3 color = ambient_color * tex_color + diffuse * light_color * tex_color;
    fragColor = vec4(color, 1.0);
}
