#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec4 outFragColor;
layout(location = 1) out float outDepth;

layout(set = 0, binding = 0) uniform sampler2D texSamplers[];

layout(push_constant) uniform PC {
    int albedo;
} pc;

void main() {
    vec4 albedoColor = texture(texSamplers[pc.albedo], inTexCoord);
    outFragColor = inColor * albedoColor;
    gl_FragDepth = gl_FragCoord.z;
}

