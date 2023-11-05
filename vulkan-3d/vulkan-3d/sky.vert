#version 460
#define MAX_MODELS 1200

layout(location = 0) in vec3 inPosition; 
layout(location = 0) out vec4 outPos;


struct matrixUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
};
layout(set = 0, binding = 0) buffer matBufferObject {
    matrixUBO matrixSSBO[MAX_MODELS];
} matSSBO;



void main() {
    mat4 view = matSSBO.matrixSSBO[0].view;
    mat4 projection = matSSBO.matrixSSBO[0].proj;
    mat4 rotationOnlyView = view;
    rotationOnlyView[3] = vec4(0.0, 0.0, 0.0, 1.0);
    
    outPos = projection * rotationOnlyView * vec4(inPosition, 1.0);
    gl_Position = outPos;
}



