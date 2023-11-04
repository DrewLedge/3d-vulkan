#version 460
#define MAX_MODELS 1200

layout(location = 0) in vec3 inPosition; 


struct matrixUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
};
layout(set = 0, binding = 0) buffer matBufferObject {
    matrixUBO matrixSSBO[MAX_MODELS];
} matSSBO;



void main() {
}