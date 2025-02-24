#version 460

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(location = 0) in vec4 inFragPos; // the direction to sample from the cubemap
layout(location = 0) out vec4 outColor; // output color of the fragment

layout(set = 0, binding = 0) uniform samplerCube cubeMap;

void main() {
    outColor = texture(cubeMap, inFragPos.xyz);
    outColor.rgb *= 0.01;
}
