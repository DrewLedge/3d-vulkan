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
    mat4 lightSpace = lightProj * lightView;
    vec4 transformedPos = lightSpace * modelMatrix * vec4(inPosition, 1.0);

    transformedPos.xyz /= transformedPos.w;
    transformedPos.z = (transformedPos.z + transformedPos.w) / 2.0;
    transformedPos.y *= -1; // flip y

    gl_Position = transformedPos;

}



