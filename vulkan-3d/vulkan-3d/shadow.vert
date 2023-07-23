#version 460
#define MAX_MODELS 1200
layout (location = 0) in vec3 inPosition;

struct matrixUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
};

layout(set = 0, binding = 0) buffer matBufferObject {
    matrixUBO matrixSSBO[MAX_MODELS];
} matSSBO;

struct light {
    mat4 viewMatrix;
    mat4 projectionMatrix;
};

layout(set = 1, binding = 3) buffer LightBuffer {
	light lights[];
};

layout(push_constant) uniform PC {
    int modelIndex;
    int lightIndex;
} pc;

void main() {
    mat4 modelMatrix = matSSBO.matrixSSBO[pc.modelIndex].model; // model matrix of the object
    mat4 mvp = lights[pc.lightIndex].projectionMatrix * lights[pc.lightIndex].viewMatrix * modelMatrix; // model view proj matrix
    vec4 lightSpacePos = mvp * vec4(inPosition, 1.0); 
    lightSpacePos = lightSpacePos / lightSpacePos.w; // perspective divide
    gl_Position = lightSpacePos;
}



