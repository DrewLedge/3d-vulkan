#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 1) uniform sampler2D texSamplers[];

layout(location = 0) in vec4 fragColor;
layout(location = 2) in float inAlpha;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) flat in uint inVertInex;
layout(location = 5) flat in uint inTexInex;
layout(location = 6) flat in uint inModelInex;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 sampled = texture(texSamplers[0], inTexCoord);
    outColor = sampled * vec4(fragColor.rgb, inAlpha);
}












