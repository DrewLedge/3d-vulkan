#version 460

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec4 outFragColor;
layout(location = 1) out float outDepth;

void main() {
    vec4 finalColor = inColor;
    outFragColor = finalColor;
    outDepth = gl_FragCoord.z;
}

