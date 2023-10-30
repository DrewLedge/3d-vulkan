#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define MAX_TEXTURES 4000 // this number must be the same as in the main code!!!
#define MAX_MODELS 1200

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec3 inColor; 
layout(location = 2) in float inAlpha;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uint inVertIndex;
layout(location = 5) in vec3 inNormal;
layout(location = 6) in vec4 inTangent;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 outTexCoord;
layout(location = 2) flat out uint outTexIndex;
layout(location = 3) flat out uint outModelIndex;
layout(location = 4) out vec3 outFragPos;
layout(location = 5) out vec3 outViewDirection;
layout(location = 6) out mat3 TBN;
layout(location = 20) out vec3 test;



struct matrixUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
};
layout(set = 0, binding = 0) buffer matBufferObject {
    matrixUBO matrixSSBO[MAX_MODELS];
} matSSBO;

layout(set = 2, binding = 2) buffer BufferObject {
    uint texIndices[MAX_TEXTURES];     
    uint modelIndices[MAX_MODELS]; // which materials are part of which objects. (add functionality for this)
} idxSSBO;

void main() {
    uint modelIndex = inVertIndex;
    uint texIndex = inVertIndex*3; // 3 textures per material

    mat4 proj = matSSBO.matrixSSBO[modelIndex].proj;
    mat4 view = matSSBO.matrixSSBO[modelIndex].view;
    mat4 model = matSSBO.matrixSSBO[modelIndex].model;

    vec3 worldCamPos = vec3(inverse(view)[3]);
    vec4 worldPos = model * vec4(inPosition, 1.0);
    vec3 viewDir = normalize(worldPos.xyz - worldCamPos);

    gl_Position = proj * view * model * vec4(inPosition, 1.0);
    fragColor = vec4(inColor, 1.0);
    outTexCoord = inTexCoord;
    if (texIndex <= MAX_TEXTURES) {
        outTexIndex = texIndex; // pass the texture index to the fragment shader
    }
    if (modelIndex <= MAX_MODELS) {
        outModelIndex = modelIndex; // pass the model/material index to the fragment shader
    }

    vec3 normal = normalize(mat3(model) * inNormal);
    vec3 tangent = normalize(mat3(model) * inTangent.xyz);
    vec3 bitangent = cross(normal, tangent) * inTangent.w;  // handedness is in inTangent.w
    TBN = mat3(tangent, bitangent, normal);
    test = bitangent * 0.5 + 0.5;


    //for lighting
    outFragPos = vec3(model * vec4(inPosition, 1.0)); // position in world space
    outViewDirection = viewDir;

}
