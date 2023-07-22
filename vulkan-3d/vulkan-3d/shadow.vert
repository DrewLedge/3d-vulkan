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
    mat4 modelMatrix = matSSBO.matrixSSBO[pc.modelIndex].model;
    gl_Position = lights[pc.lightIndex].projectionMatrix * lights[pc.lightIndex].viewMatrix * modelMatrix * vec4(inPosition, 1.0);
}



