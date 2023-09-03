#version 460
#define MAX_MODELS 1200
layout (location = 0) in vec3 inPosition;

struct matrixUBO {
    mat4 model;
};

layout(set = 0, binding = 0) buffer matBufferObject {
    matrixUBO matrixSSBO[MAX_MODELS];
} matSSBO;

struct light {
    mat4 viewMatrix;
    mat4 projectionMatrix;
};

layout(set = 1, binding = 3) buffer LightBuffer {
	light lights[];
};

layout(push_constant) uniform PC {
    int modelIndex;
    int lightIndex;
} pc;

void main() {
	// get the model matrix of the model from the SSBO
    mat4 modelMatrix = matSSBO.matrixSSBO[pc.modelIndex].model; // model matrix of the model
    mat4 lightView = lights[pc.lightIndex].viewMatrix;
    mat4 lightProj = lights[pc.lightIndex].projectionMatrix;
    mat4 lightSpaceMatrix = lightProj * lightView;
    
    // transform the vertex into light space
    gl_Position = lightSpaceMatrix * modelMatrix * vec4(inPosition, 1.0);
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0; // get the right depth val
    gl_Position.y *= -1; // flip the y axis for vulkan
}



