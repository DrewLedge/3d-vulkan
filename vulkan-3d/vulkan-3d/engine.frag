#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 fragColor;
layout(location = 2) in vec2 inAlpha;

layout(location = 0) out vec4 outColor;

void main() {
   outColor = vec4(fragColor.rgb, inAlpha.x);
}







