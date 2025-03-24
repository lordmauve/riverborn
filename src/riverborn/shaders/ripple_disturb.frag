#version 330
uniform vec2 p1;         // start point of line in texture coords
uniform vec2 p2;         // end point of line in texture coords
uniform float thickness; // effective thickness of the line
uniform float intensity; // disturbance strength to add
in vec2 v_texcoord;
out vec4 fragColor;
void main() {
    // Compute distance from the current fragment to the line segment p1-p2.
    vec2 pa = v_texcoord - p1;
    vec2 ba = p2 - p1;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    float dist = length(pa - ba * h);
    // Smoothly add an impulse if within the thickness.
    float effect = (1.0 - smoothstep(0.0, thickness, dist)) * intensity;
    fragColor = vec4(effect, 0.0, 0.0, 1.0);
}
