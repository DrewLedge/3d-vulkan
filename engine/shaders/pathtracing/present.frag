#version 460

#extension GL_EXT_nonuniform_qualifier : require

layout(push_constant, std430) uniform pc {
    int frame;
};

layout(set = 0, binding = 0, rgba16f) uniform readonly image2D rtTextures[];
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
    ivec2 p = ivec2(vec2(imageSize(rtTextures[frame])) * inUV);
    outColor = imageLoad(rtTextures[frame], p);
}
