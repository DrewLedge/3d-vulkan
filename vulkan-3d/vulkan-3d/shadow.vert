#version 460

layout (location = 0) in vec3 inPosition;

layout (binding = 0) uniform UBO {
    mat4 lightViewProj; // light space view projection matrix
} ubo;

layout (location = 0) out vec4 fragPosLightSpace;

void main() {
    fragPosLightSpace = ubo.lightViewProj * vec4(inPosition, 1.0);
    gl_Position = fragPosLightSpace;
}

