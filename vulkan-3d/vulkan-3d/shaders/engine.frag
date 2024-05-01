#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 0) uniform sampler2D texSamplers[];
layout(set = 2, binding = 2) uniform sampler2DShadow shadowMapSamplers[];

layout(location = 0) in vec4 inFragColor;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) flat in uint inTexIndex;
layout(location = 3) in vec3 inFragPos;
layout(location = 4) in vec3 inViewDir;
layout(location = 5) in mat3 inTBN;
layout(location = 8) flat in uint inRender;
layout(location = 9) flat in uint inBitfield;

layout(location = 0) out vec4 outColor; 

struct LightData {
    mat4 view;
    mat4 proj;

    vec4 pos;
    vec4 color;
    vec4 targetVec;
    float intensity;
	float innerConeAngle; // in degrees
	float outerConeAngle; // in degrees
	float constantAttenuation;
	float linearAttenuation;
	float quadraticAttenuation;
};

layout (set=1, binding = 1) buffer LightBuffer {
    LightData lights[];
};

vec4 albedo = vec4(0.0f);
vec4 metallicRoughness = vec4(1.0f);
vec3 normal = vec3(1.0f);
vec3 emissive = vec3(0.0f);
float occlusion = 1.0f;

#include "includes/fragformulas.glsl"

void main() {
    if (inRender == 1) {
        discard;
    }

    getTextures();
    outColor = calcLighting(true, false, 0.005f);
}

