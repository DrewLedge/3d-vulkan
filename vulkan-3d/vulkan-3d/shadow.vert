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
    mat4 view;
    mat4 projection;
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
    mat4 lightView = lights[pc.lightIndex].view;
    mat4 lightProj = lights[pc.lightIndex].projection;
    mat4 modelMatrix = matSSBO.matrixSSBO[pc.modelIndex].model; // model matrix of the model
    vec4 transformedPos = lightProj * lightView * modelMatrix * vec4(inPosition, 1.0);
    transformedPos.z = (transformedPos.z + transformedPos.w) / 2.0; // depth range adjustment for vulkan
    transformedPos.y *= -1; // flip y

    gl_Position = transformedPos;

}



