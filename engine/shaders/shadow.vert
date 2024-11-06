#version 460

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
};

layout(push_constant) uniform PC {
    int lightIndex;
} pc;

void main() {
    int index = pc.lightIndex;

    // fetch matrices
    mat4 model = mat4(inModel1, inModel2, inModel3, inModel4); // model matrix of the model
    gl_Position = getPos(lights[index].proj, lights[index].view, model, inPosition); // transform position
}
