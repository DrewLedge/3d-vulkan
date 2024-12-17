#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 0) uniform sampler2D texSamplers[];

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

layout(set = 2, binding = 0) uniform sampler2DShadow shadowMapSamplers[];

layout(set = 3, binding = 0) uniform CamBufferObject {
    mat4 view;
    mat4 proj;
    mat4 iview;
    mat4 iproj;
} CamUBO[];

layout(set = 4, binding = 0) uniform sampler2D depthSampler[];

layout(location = 0) in vec2 inTexCoord;
layout(location = 1) flat in int inFrame;

layout(location = 0) out vec4 outColor;

#include "includes/fragformulas.glsl"
#include "includes/lighting.glsl"

layout(push_constant, std430) uniform PC {
    layout(offset = 4) int lightCount;
    int frameCount;
};

void main() {
    float depth = texture(depthSampler[inFrame], inTexCoord).r;
    if (depth == 1.0f) discard;

    // load textures
    int baseIndex = inFrame * 4;
    vec4 albedo = texture(texSamplers[baseIndex], inTexCoord);
    vec4 metallicRoughness = texture(texSamplers[baseIndex + 1], inTexCoord);
    vec3 normal = texture(texSamplers[baseIndex + 2], inTexCoord).rgb;
    vec3 emissive = texture(texSamplers[baseIndex + 3], inTexCoord).rgb;
    float occlusion = texture(texSamplers[baseIndex + 3], inTexCoord).a;

    // convert to -1 to 1 range
    normal = normal * 2.0f - 1.0f;

    // get the pos and view
    vec3 fragPos = getFragPos(inTexCoord, depth, CamUBO[inFrame].iproj, CamUBO[inFrame].iview);
    vec3 viewDir = getViewDir(fragPos, CamUBO[inFrame].iview);

    // calc lighting on the fragment
    outColor = calcLighting(albedo, metallicRoughness, normal, emissive, occlusion, fragPos, viewDir, inFrame, frameCount, lightCount, false, false);
}
