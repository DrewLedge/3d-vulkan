#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) out vec2 outTexCoord;
layout(location = 1) flat out int outFrame;

layout(push_constant, std430) uniform PC {
    int frame;
};

vec2 positions[6] = {
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2(1.0,  1.0),
    vec2(-1.0,  1.0)
};

vec2 uvs[6] = {
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
};

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    outTexCoord = uvs[gl_VertexIndex];
    outFrame = frame;
}
