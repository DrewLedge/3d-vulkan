#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 5) uniform sampler2D textures[];

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    int numPeels;
} pc;

void main() {
    // get the color and depth from the main pass 
    vec4 mainColor = texture(textures[0], inUV);
    float mainDepth = texture(textures[1], inUV).r;

    // get the weighted color and alpha from the wboit pass
    vec4 weightedColor = texture(textures[2], inUV);
    float weightedAlpha = texture(textures[3], inUV).r;

    outColor = weightedColor;

}
