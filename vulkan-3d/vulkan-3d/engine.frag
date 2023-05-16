#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
   outColor = vec4(fragColor.rgb, 0.6); // 60% transparency!
}




