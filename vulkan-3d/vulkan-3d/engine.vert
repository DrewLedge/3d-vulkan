#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec3 inColor; 
layout(location = 2) in float inAlpha;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uint inVertIndex;

layout(location = 0) out vec4 fragColor;
layout(location = 2) out float outAlpha;
layout(location = 3) out vec2 outTexCoord;
layout(location = 4) flat out uint outVertIndex;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = vec4(inColor, 1.0);
    outAlpha = inAlpha;
    outTexCoord = inTexCoord;
    outVertIndex = inVertIndex;
}









