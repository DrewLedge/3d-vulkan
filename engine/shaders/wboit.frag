#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 0) uniform sampler2D texSamplers[];

layout(set = 2, binding = 0) uniform sampler2DShadow shadowMapSamplers[];

layout(set = 4, binding = 0) uniform sampler2D depthSamplers[];

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

layout(location = 0) in vec2 inTexCoord;
layout(location = 1) in vec3 inFragPos;
layout(location = 2) in vec3 inViewDir;
layout(location = 3) in mat3 inTBN; // uses locations 3, 4 and 5
layout(location = 6) flat in uint inRender;
layout(location = 7) in float inFarPlane;
layout(location = 8) in float inNearPlane;
layout(location = 9) flat in int inFrame;

layout(location = 0) out vec4 outColor;

layout(push_constant, std430) uniform pc {
    layout(offset = 4) int lightCount;
    int frameCount;
    int bitfield;
    int texInd;
};

vec4 albedo = vec4(1.0f);
vec4 metallicRoughness = vec4(0.0f, 0.5f, 0.0f, 1.0f);
vec3 normal = vec3(0.0f);
vec3 emissive = vec3(0.0f);
float occlusion = 1.0f;

#include "includes/fragformulas.glsl"
#include "includes/lighting.glsl"

float getWeight(float z, float a) {
    float weight = a * exp(-z);
    return 1.0 - weight;
}

void main() {
    if (inRender == 1) {
        discard;
    }

    getTextures(bitfield, texInd, inTexCoord, inTBN);

    vec4 color = calcLighting(inFrame, lightCount, false, true);

    // get the depth from the opaque texture
    vec2 cords = getTexCords(depthSamplers[inFrame], gl_FragCoord.xy);
    float oDepth = texture(depthSamplers[inFrame], cords).r;
    oDepth = linDepth(oDepth, inNearPlane, inFarPlane);

    // get the depth of the fragment
    float tDepth = linDepth(gl_FragCoord.z, inNearPlane, inFarPlane);

    // if the transparent depth is greater than the opaque depth, discard
    if (tDepth > oDepth) discard;

    // get the weight and output the color and alpha
    float weight = getWeight(gl_FragCoord.z, color.a);
    outColor = vec4(color.rgb * weight, weight);
}
