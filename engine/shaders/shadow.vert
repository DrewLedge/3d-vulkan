#version 460

#extension GL_EXT_nonuniform_qualifier : enable

#include "includes/vertformulas.glsl"

layout(location = 0) in vec3 inPosition;

// individual rows of the instanced model matrix
layout(location = 1) in vec4 inModel1;
layout(location = 2) in vec4 inModel2;
layout(location = 3) in vec4 inModel3;
layout(location = 4) in vec4 inModel4;

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

layout(set = 0, binding = 0) readonly buffer LightBuffer {
    LightData lights[];
} lssbo[];

layout(push_constant, std430) uniform pc {
    int frame;
    int lightIndex;
};

void main() {
    mat4 model = mat4(inModel1, inModel2, inModel3, inModel4); // model matrix of the model
    gl_Position = getPos(lssbo[frame].lights[lightIndex].proj, lssbo[frame].lights[lightIndex].view, model, inPosition); // transform position
}
