#version 460

#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 inPos;
layout(location = 0) out vec4 outPos;

layout(push_constant, std430) uniform PC {
    int frame;
};

layout(set = 1, binding = 0) uniform CamBufferObject {
    mat4 view;
    mat4 proj;
    mat4 iview;
    mat4 iproj;
} CamUBO[];

void main() {
    mat4 view = CamUBO[frame].view;
    mat4 projection = CamUBO[frame].proj;
    mat4 rotationOnlyView = view;
    rotationOnlyView[3] = vec4(0.0f, 0.0f, 0.0f, 1.0f);

    outPos = vec4(inPos, 1);
    gl_Position = projection * rotationOnlyView * vec4(inPos, 1.0f);
}
