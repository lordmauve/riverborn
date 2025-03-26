#version 330
in vec3 normal;
in vec2 uv;
in vec3 pos;

uniform sampler2D texture0;
uniform vec3 sun_dir;

out vec4 f_color;
const float ambient = 0.5;

void main() {
    vec4 diffuse = texture(texture0, uv);
    if (diffuse.a < 0.3) discard;

    vec3 N = normalize(normal);
    vec3 L = normalize(sun_dir);
    vec3 V = normalize(-pos);
    vec3 R = reflect(-L, N);

    float diff = clamp(dot(N, L), 0.0, 1.0);
    float spec = pow(max(dot(V, R), 0.0), 16.0);

    vec3 color = diff * diffuse.rgb + spec * vec3(1.0);
    f_color = vec4(color, 1.0);
}
