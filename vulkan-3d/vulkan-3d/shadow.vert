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

struct light {
    mat4 view;
    mat4 projection;
};

layout(set = 1, binding = 3) buffer LightBuffer {
	light lights[];
};

layout(push_constant) uniform PC {
    int modelIndex;
    int lightIndex;
} pc;

void main() {
    // fetch matrices
    mat4 lightView = lights[pc.lightIndex].view;
    mat4 lightProj = lights[pc.lightIndex].projection;
    mat4 modelMatrix = matSSBO.matrixSSBO[pc.modelIndex].model; // model matrix of the model

    // transform position
    mat4 lightSpace = lightProj * lightView;
    vec4 transformedPos = lightSpace * modelMatrix * vec4(inPosition, 1.0);

    gl_Position = transformedPos;

}




