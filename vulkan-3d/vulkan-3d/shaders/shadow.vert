#version 460
#define MAX_MODELS 1200

#include "includes/vertformulas.glsl"

layout (location = 0) in vec3 inPosition;

// individual rows of the instanced model matrix
layout(location = 1) in vec4 model1;
layout(location = 2) in vec4 model2;
layout(location = 3) in vec4 model3;
layout(location = 4) in vec4 model4;

struct lightMatrix {
    mat4 viewMatrix;
    mat4 projectionMatrix;
};
struct lightData {
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

layout (set=0, binding = 1) buffer LightBuffer {
	lightMatrix lightMatricies[20]; // are used in the shader
    lightData lights[20]; // not used in this shader but needed because of the descriptor set and binding consistancy
};

layout(push_constant) uniform PC {
    int modelIndex;
    int lightIndex;
} pc;

void main() {
    // fetch matrices
    mat4 lightView = lightMatricies[pc.lightIndex].viewMatrix;
    mat4 lightProj = lightMatricies[pc.lightIndex].projectionMatrix;
    mat4 model = mat4(model1, model2, model3, model4); // model matrix of the model

    gl_Position = getPos(lightProj, lightView, model, inPosition); // transform position
}




