#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define MAX_TEXTURES 1000 // this number must be the same as in the main code!!!
#define MAX_MODELS 300

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec3 inColor; 
layout(location = 2) in float inAlpha;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uint inVertIndex;

layout(location = 0) out vec4 fragColor;
layout(location = 2) out float outAlpha;
layout(location = 3) out vec2 outTexCoord;
layout(location = 4) flat out uint outVertIndex;
layout(location = 5) flat out uint outTexIndex;
layout(location = 6) flat out uint outModelIndex;

struct matrixUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
};
layout(set = 0, binding = 0) buffer matBufferObject {
    matrixUBO matrixSSBO[MAX_MODELS];
} matSSBO;

layout(set = 2, binding = 2) buffer BufferObject {
    uint texIndices[MAX_TEXTURES]; // which textures are used by the model (1,1,1 would be model 1)
    uint modelIndices[MAX_MODELS]; // which materials are part of which objects. (3,3,3 would be object 3)
} idxSSBO;

void main() {
    uint modelIndex = idxSSBO.modelIndices[inVertIndex];
    uint texIndex = idxSSBO.texIndices[inVertIndex];

    mat4 proj = matSSBO.matrixSSBO[modelIndex].proj;
    mat4 view = matSSBO.matrixSSBO[modelIndex].view;
    mat4 model = matSSBO.matrixSSBO[modelIndex].model;

    gl_Position = proj*view*model * vec4(inPosition, 1.0);
    fragColor = vec4(inColor, 1.0);
    outAlpha = inAlpha;
    outTexCoord = inTexCoord;

    if (texIndex <= MAX_TEXTURES) {
    outTexIndex = texIndex; // pass the texture index to the fragment shader
    }
     if (modelIndex <= MAX_MODELS) {
    outModelIndex = modelIndex; // pass the model/material index to the fragment shader
    }
}









