#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "includes/vertformulas.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;

// per-instance data
layout(location = 4) in vec4 inModel1;
layout(location = 5) in vec4 inModel2;
layout(location = 6) in vec4 inModel3;
layout(location = 7) in vec4 inModel4;
layout(location = 8) in uint inToRender;

layout(location = 0) out vec2 outTexCoord;
layout(location = 1) flat out uint outTexIndex;
layout(location = 2) out vec3 outFragPos;
layout(location = 3) out vec3 outViewDir;
layout(location = 4) out mat3 outTBN; // uses locations 4, 5 and 6
layout(location = 7) flat out uint outRender;
layout(location = 8) flat out uint outBitfield;

layout(set = 3, binding = 0) uniform CamBufferObject {
    mat4 view;
    mat4 proj;
} CamUBO;

layout(push_constant) uniform PC {
    int textureExist;
    int texInd;
} pc;

void main() {
    outRender = inToRender;
    uint texIndex = pc.texInd;
    outBitfield = pc.textureExist;

    mat4 proj = CamUBO.proj;
    mat4 view = CamUBO.view;
    mat4 model = mat4(inModel1, inModel2, inModel3, inModel4);

    vec3 viewDir = getViewDir(view, model, inPosition);
    gl_Position = getPos(proj, view, model, inPosition);
    outTBN = getTBN(inTangent, model, inNormal);

    outTexCoord = inTexCoord;
    outTexIndex = texIndex;
    outFragPos = vec3(model * vec4(inPosition, 1.0));
    outViewDir = viewDir;
}
