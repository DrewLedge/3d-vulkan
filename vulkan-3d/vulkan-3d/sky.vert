#version 460
#define MAX_MODELS 1200

layout(location = 0) in vec3 inPosition; 
layout(location = 0) out vec4 outPos;

layout(set = 1, binding = 5) uniform camBufferObject {
    mat4 view;
    mat4 proj;
} camUBO;


void main() {
    mat4 view = camUBO.view;
    mat4 projection = camUBO.proj;
    mat4 rotationOnlyView = view;
    rotationOnlyView[3] = vec4(0.0, 0.0, 0.0, 1.0);
    
    outPos = vec4(inPosition, 1);
    gl_Position = projection * rotationOnlyView * vec4(inPosition, 1.0);
}



