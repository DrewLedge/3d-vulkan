#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) out vec2 outUV;

vec2 positions[6] = {
    vec2(-1.0f, -1.0f),
    vec2(1.0f, -1.0f),
    vec2(1.0f,  1.0f),
    vec2(-1.0f, -1.0f),
    vec2(1.0f,  1.0f),
    vec2(-1.0f,  1.0f)
};

vec2 uvs[6] = {
    vec2(0.0f, 0.0f),
    vec2(1.0f, 0.0f),
    vec2(1.0f, 1.0f),
    vec2(0.0f, 0.0f),
    vec2(1.0f, 1.0f),
    vec2(0.0f, 1.0f)
};

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0f, 1.0f);
    outUV = uvs[gl_VertexIndex];
}
