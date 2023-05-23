#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_KHR_vulkan_glsl : enable
layout(set = 0, binding = 0) uniform sampler2D texSampler; // The texture sampler

layout(location = 0) in vec4 fragColor;
layout(location = 2) in float inAlpha;
layout(location = 3) in vec2 inTexCoord; // Input texture coordinates from vertex shader

layout(location = 0) out vec4 outColor;

void main() {
    vec4 sampled = texture(texSampler, inTexCoord); // Sample the texture
    outColor = sampled * vec4(fragColor.rgb, inAlpha); // Multiply by incoming color and alpha
}









