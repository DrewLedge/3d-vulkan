#version 460

layout (location = 0) in vec3 inPosition;

struct light {
    vec3 lPos;
    vec3 lColor;
    float lightIntensity;
    mat4 viewMatrix;
    mat4 modelMatrix;
    mat4 projectionMatrix;
};

layout (set=3, binding = 3) buffer LightBuffer {
	light lights[];
};

layout(push_constant) uniform PC {
    int lightIndex;
} pc;

void main() {
    gl_Position = lights[pc.lightIndex].viewMatrix* lights[pc.lightIndex].projectionMatrix* vec4(inPosition, 1.0);
}


