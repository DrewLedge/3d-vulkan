#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define MAX_TEXTURES 100 // this number must be the same as in the main code!!!
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

layout(set = 0, binding = 0) uniform UniformBufferObjects {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo[];

layout(std430, binding = 2) buffer BufferObject {
    uint texIndices[MAX_TEXTURES]; // which textures are used by the model (1,1,1 would be model 1)
    uint modelIndices[MAX_MODELS]; // which models are part of which objects. (3,3,3 would be object 3)
    uint texIndicesCount;
	uint modelIndicesCount;
} ssbo;

void main() {
    uint modelIndex = ssbo.modelIndices[inVertIndex];
    uint texIndex = ssbo.texIndices[inVertIndex];
    gl_Position = ubo[modelIndex].proj * ubo[modelIndex].view * ubo[modelIndex].model * vec4(inPosition, 1.0);
    fragColor = vec4(inColor, 1.0);
    outAlpha = inAlpha;
    outTexCoord = inTexCoord;
    outVertIndex = inVertIndex; // pass the texture index to the fragment shader
    outTexIndex = texIndex; // pass the texture index to the fragment shader
    outModelIndex = modelIndex; // pass the model index to the fragment shader
}









