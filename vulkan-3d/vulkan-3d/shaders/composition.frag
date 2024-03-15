#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 6) uniform sampler2D depthPeels[];
layout(set = 1, binding = 7) uniform sampler2D mainPass;

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    int numPeels;
} pc;

void main() {
	vec4 mainColor = texture(mainPass, inUV);
	outColor = mainColor;
}



