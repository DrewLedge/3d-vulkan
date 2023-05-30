#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition; // 3D position of vertex
layout(location = 1) in vec3 inColor; // color of vertex
layout(location = 2) in float inAlpha; // alpha value of vertex
layout(location = 3) in vec2 inTexCoord; // Texture coordinates

layout(location = 0) out vec4 fragColor; // output color of vertex shader
layout(location = 2) out float outAlpha;
layout(location = 3) out vec2 outTexCoord; // output texture coordinates to fragment shader

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo; // UBO declaration

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = vec4(inColor, 1.0);
    outAlpha = inAlpha;
    outTexCoord = inTexCoord;
}







