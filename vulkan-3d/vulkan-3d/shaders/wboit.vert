#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define MAX_TEXTURES 4000 // this number must be the same as in the main code!!!

#include "includes/vertformulas.glsl"

layout(location = 0) in vec3 inPosition; 
layout(location = 1) in vec4 inColor; 
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;

// per-instance data
layout(location = 5) in vec4 model1;
layout(location = 6) in vec4 model2;
layout(location = 7) in vec4 model3;
layout(location = 8) in vec4 model4;
layout(location = 9) in uint toRender;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 outTexCoord;
layout(location = 2) flat out uint outTexIndex;
layout(location = 3) out vec3 outFragPos;
layout(location = 4) out vec3 outViewDirection;
layout(location = 5) out mat3 TBN;
layout(location = 8) flat out uint render; // if 0 render, if 1 don't render
layout(location = 9) flat out uint bitfield; // number of textures per model
layout(location = 10) out float farPlane;
layout(location = 11) out float nearPlane;

layout(set = 3, binding = 4) uniform camBufferObject {
    mat4 view;
    mat4 proj;
} camUBO;

layout(push_constant) uniform PC {
    int textureExist;
    int texInd;
} pc;

void main() {
    render = toRender;
    uint texIndex = pc.texInd;
    bitfield = pc.textureExist;

    mat4 proj = camUBO.proj;
    mat4 view = camUBO.view;
    mat4 model = mat4(model1, model2, model3, model4);

    vec3 viewDir = getViewDir(view, model, inPosition);
    gl_Position = getPos(proj, view, model, inPosition);
    TBN = getTBN(inTangent, model, inNormal);

    fragColor = inColor;
    outTexCoord = inTexCoord;
    if (texIndex <= MAX_TEXTURES) outTexIndex = texIndex; // pass the texture index to the fragment shader
    outFragPos = vec3(model * vec4(inPosition, 1.0)); // position in world space
    outViewDirection = viewDir;

    farPlane = getFarPlane(proj);
    nearPlane = getNearPlane(proj);
}
