#version 460

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec4 outPos;

layout(set = 1, binding = 0) uniform CamBufferObject {
    mat4 view;
    mat4 proj;
} CamUBO;


void main() {
    mat4 view = CamUBO.view;
    mat4 projection = CamUBO.proj;
    mat4 rotationOnlyView = view;
    rotationOnlyView[3] = vec4(0.0, 0.0, 0.0, 1.0);

    outPos = vec4(inPos, 1);
    gl_Position = projection * rotationOnlyView * vec4(inPos, 1.0);
}
