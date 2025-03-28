#version 330
in vec3 frag_normal;
in vec2 frag_uv;

uniform sampler2D diffuse_tex;
uniform vec3 light_dir;
uniform vec3 light_color;
uniform vec3 ambient_color;

out vec4 fragColor;

void main() {
    vec4 diffuse = texture(diffuse_tex, frag_uv);
    if (diffuse.a < 0.3) discard;

    vec3 normal = normalize(frag_normal);
    vec3 light = normalize(light_dir);
    float diffuseLight = max(dot(normal, light), 0.0);
    vec3 color = (ambient_color + diffuseLight * light_color) * diffuse.rgb;
    fragColor = vec4(color, 1.0);
}
