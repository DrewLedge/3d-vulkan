#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define MAX_TEXTURES 4000 // this number must be the same as in the main code!!!
#define MAX_MODELS 1200

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec4 inColor; 
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
layout(location = 9) out float handedness;
layout(location = 10) flat out uint render; // if 0 render, if 1 don't render
layout(location = 11) flat out uint bitfield; // number of textures per model

layout(location = 20) out vec3 test;



struct matrixUBO {
    mat4 model;
};
layout(set = 0, binding = 0) buffer matBufferObject {
    matrixUBO matrixSSBO[MAX_MODELS];
} matSSBO;

layout(set = 4, binding = 5) buffer camBufferObject {
    mat4 view;
    mat4 proj;
} camSSBO;

layout(push_constant) uniform PC {
    int notRender;
    int textureExist;
    int texCount;
} pc;


void main() {
    uint modelIndex = inVertIndex;
    render = 0;
    if (pc.notRender == modelIndex) {
        render = 1; // dont render this model
	}
    uint texIndex = inVertIndex * pc.texCount;
    bitfield = pc.textureExist;

    mat4 proj = camSSBO.proj;
    mat4 view = camSSBO.view;
    mat4 model = matSSBO.matrixSSBO[modelIndex].model;

    vec3 worldCamPos = vec3(inverse(view)[3]);
    vec4 worldPos = model * vec4(inPosition, 1.0);
    vec3 viewDir = normalize(worldPos.xyz - worldCamPos);

    gl_Position = proj * view * model * vec4(inPosition, 1.0);
    fragColor = inColor;
    outTexCoord = inTexCoord;
    if (texIndex <= MAX_TEXTURES) {
        outTexIndex = texIndex; // pass the texture index to the fragment shader
    }
    if (modelIndex <= MAX_MODELS) {
        outModelIndex = modelIndex; // pass the model/material index to the fragment shader
    }
    handedness = inTangent.w;
    
    mat3 normMat = transpose(inverse(mat3(model)));
    vec3 N = normalize(normMat * inNormal);
    vec3 T = normalize(normMat * inTangent.xyz);
    T = normalize(T - dot(T, N) * N); // re orthogonalize tangent
    vec3 B = normalize(cross(N, T) * handedness);  
    TBN = mat3(T, B, N);
    test = N;


    //for lighting
    outFragPos = vec3(model * vec4(inPosition, 1.0)); // position in world space
    outViewDirection = viewDir;

}
