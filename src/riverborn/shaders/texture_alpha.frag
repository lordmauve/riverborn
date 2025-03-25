#version 330

out vec4 fragColor;
uniform sampler2D texture0;

in vec3 normal;
in vec3 pos;
in vec2 uv;

void main()
{
    float l = dot(normalize(-pos), normalize(normal));
    vec4 color = texture(texture0, uv);
    if (color.a < 0.3) discard;
    fragColor = vec4(color.rgb * 0.25 + color.rgb * 0.75 * abs(l), 1.0);
}
