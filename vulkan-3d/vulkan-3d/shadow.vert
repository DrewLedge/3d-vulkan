#version 460

layout (location = 0) in vec3 inPosition;

struct light {
    mat4 viewMatrix;
    mat4 projectionMatrix;
};

layout(std430, binding = 3) buffer LightBuffer {
	light lights[];
};

layout(push_constant) uniform PC {
    int lightIndex;
} pc;

void main() {
    gl_Position = lights[pc.lightIndex].projectionMatrix* lights[pc.lightIndex].viewMatrix* vec4(inPosition, 1.0);
}


