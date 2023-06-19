#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_KHR_vulkan_glsl : enable


layout(location = 0) in vec4 fragColor;
layout(location = 2) in float inAlpha;
layout(location = 3) in vec2 inTexCoord; // Input texture coordinates from vertex shader
layout(location=4) in uint inVertInex; // an int of which vertex goes to which material
uniform sampler2D texSampler;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 sampled = texture(texSampler, inTexCoord); // sample the texture with automatic mip level selection
    outColor = sampled * vec4(fragColor.rgb, inAlpha);
}











