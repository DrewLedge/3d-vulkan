#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 0) uniform sampler2D deferredTextures[];
layout(set = 2, binding = 0) uniform sampler2DShadow shadowMapSamplers[];

layout(location = 0) in vec2 inTexCoord;
layout(location = 0) out vec4 outColor;

struct LightData {
    vec4 pos;
    vec4 color;
    vec4 target;

    mat4 proj;
    mat4 view;

    float intensity;
    float innerConeAngle;
    float outerConeAngle;
    float constantAttenuation;
    float linearAttenuation;
    float quadraticAttenuation;
};

layout(set = 1, binding = 0) readonly buffer LightBuffer {
    LightData lights[];
} lssbo[];

layout(set = 4, binding = 0) uniform sampler2D depthSampler[];

layout(push_constant) uniform PC {
    int frame;
};

void main() {
    int baseIndex = frame * 4;
    vec4 albedo = texture(deferredTextures[baseIndex], inTexCoord);

    float depth = texture(depthSampler[frame], inTexCoord).r;
    if (depth == 1.0f) discard;
    outColor = albedo;
}

