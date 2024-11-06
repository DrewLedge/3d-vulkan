#version 460

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D rtTexture;
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
    ivec2 p = ivec2(vec2(imageSize(rtTexture)) * inUV);
    outColor = imageLoad(rtTexture, p);
}
