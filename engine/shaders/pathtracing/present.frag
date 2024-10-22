#version 460

layout(set = 0, binding = 8) uniform sampler2D rtTexture;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 color = texture(rtTexture, inUV);
    outColor = vec4(color.xyz, 1.0);
}
