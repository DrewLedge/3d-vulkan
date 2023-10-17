#version 460
#define MAX_MODELS 1200
layout (location = 0) in vec3 inPosition;

struct matrixUBO { // matricies of the 3d models
    mat4 model;
    mat4 modelView; // not used
    mat4 modelProj; // not used
};

layout(set = 0, binding = 0) buffer matBufferObject {
    matrixUBO matrixSSBO[MAX_MODELS];
} matSSBO;

struct formsVec3 { // custom structure to hold my vec3s
    float x;
    float y;
    float z;
};

struct lightMatrix {
    mat4 viewMatrix;
    mat4 projectionMatrix;
};
struct lightData {
    formsVec3 pos;
    formsVec3 color;
    formsVec3 targetVec;
    float intensity;
	float innerConeAngle; // in degrees
	float outerConeAngle; // in degrees
	float constantAttenuation;
	float linearAttenuation;
	float quadraticAttenuation;
};


layout (set=1, binding = 3) buffer LightBuffer {
	lightMatrix lightMatricies[20];
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
    mat4 modelMatrix = matSSBO.matrixSSBO[pc.modelIndex].model; // model matrix of the model

    // transform position
    mat4 lightSpace = lightProj * lightView;
    vec4 transformedPos = lightSpace * modelMatrix * vec4(inPosition, 1.0);

    gl_Position = transformedPos;

}




