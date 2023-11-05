#version 460

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(location = 0) in vec4 fragPos; // the direction to sample from the cubemap
layout(location = 0) out vec4 outColor; // output color of the fragment

layout(set = 1, binding = 5) uniform samplerCube cubeMap;

void main() {
    outColor = vec4(0.4,0.6,0.2,1);
}



