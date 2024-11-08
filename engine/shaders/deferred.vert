#version 460
#extension GL_EXT_nonuniform_qualifier : require

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
layout(location = 2) out mat3 outTBN; // uses locations 2, 3 and 4
layout(location = 5) flat out uint outRender;
layout(location = 6) flat out uint outBitfield;

layout(set = 1, binding = 0) uniform CamBufferObject {
    mat4 view;
    mat4 proj;
} CamUBO;

layout(push_constant) uniform PC {
    int textureExist;
    int texInd;
} pc;

void main() {
    outRender = inToRender;
    outBitfield = pc.textureExist;

    mat4 proj = CamUBO.proj;
    mat4 view = CamUBO.view;
    mat4 model = mat4(inModel1, inModel2, inModel3, inModel4);

    gl_Position = getPos(proj, view, model, inPosition);

    outTBN = getTBN(inTangent, model, inNormal);

    outTexCoord = inTexCoord;
    outTexIndex = pc.texInd;
}
