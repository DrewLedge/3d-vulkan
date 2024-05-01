#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 0) uniform sampler2D texSamplers[];

layout(set = 2, binding = 2) uniform sampler2DShadow shadowMapSamplers[];

layout(set = 4, binding = 6) uniform sampler2D depthSampler;

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

layout(location = 0) in vec4 inFragColor;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) flat in uint inTexIndex;
layout(location = 3) in vec3 inFragPos;
layout(location = 4) in vec3 inViewDir;
layout(location = 5) in mat3 inTBN;
layout(location = 8) flat in uint inRender;
layout(location = 9) flat in uint inBitfield;
layout(location = 10) in float inFarPlane;
layout(location = 11) in float inNearPlane;

layout(location = 0) out vec4 outColor; 
layout(location = 1) out vec4 outAlpha; 

vec4 albedo = vec4(0.0f);
vec4 metallicRoughness = vec4(1.0f);
vec3 normal = vec3(1.0f);
vec3 emissive = vec3(0.0f);
float occlusion = 1.0f;

#include "includes/fragformulas.glsl"

float getWeight(float z, float a) {
    float af = pow(a + 0.01, 2.0); //alpha factor
    float df = 1.0 / (1e-5 + pow(abs(z) / 10.0, 2.0) + pow(abs(z) / 200.0, 4.0)); // depth factor
    
    float weight = af * df;
    return clamp(weight, 0.0, 2.0);
}

void main() {
    if (inRender == 1) {
        discard;
    }

    getTextures();

    vec4 color = calcLighting(false, true, 0.005f); // calculate lighting on the fragment

    // get the depth from the opaque texture
    vec2 cords = getTexCords(depthSampler);
    float oDepth = texture(depthSampler, cords).r;
    oDepth = linDepth(oDepth, inNearPlane, inFarPlane);

    // get the depth of the fragment
    float tDepth = gl_FragCoord.z;
    tDepth = linDepth(tDepth, inNearPlane, inFarPlane);

    // if the transparent depth is greater than the opaque depth, discard
    if (tDepth > oDepth) {
        discard;
    }
    
    // get the weight and output the color and alpha
    float weight = getWeight(gl_FragCoord.z, color.a);
    outColor = vec4(color.rgb * color.a * weight, color.a);
    outAlpha = vec4(color.a * weight);
}
