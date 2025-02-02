#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 0) uniform sampler2D texSamplers[];
layout(set = 2, binding = 0) uniform sampler2DShadow shadowMapSamplers[];

layout(location = 0) in vec2 inTexCoord;
layout(location = 1) flat in uint inTexIndex;
layout(location = 2) in vec3 inFragPos;
layout(location = 3) in vec3 inViewDir;
layout(location = 4) in mat3 inTBN; // uses locations 4, 5 and 6
layout(location = 7) flat in uint inRender;
layout(location = 8) flat in uint inBitfield;

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
};

vec4 albedo = vec4(1.0f);
vec4 metallicRoughness = vec4(0.0f, 0.5f, 0.0f, 1.0f);
vec3 normal = vec3(0.0f);
vec3 emissive = vec3(0.0f);
float occlusion = 1.0f;

#include "includes/fragformulas.glsl"
#include "includes/lighting.glsl"

void main() {
    if (inRender == 1) {
        discard;
    }

    getTextures(inBitfield, inTexIndex, inTexCoord, inTBN);
    outColor = calcLighting(true, false);
}

