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
layout(location = 1) out mat3 outTBN; // uses locations 1, 2 and 3
layout(location = 4) flat out uint outRender;

layout(push_constant, std430) uniform pc {
    int frame;
};

layout(set = 1, binding = 0) uniform CamBufferObject {
    mat4 view;
    mat4 proj;
} CamUBO[];

void main() {
    outRender = inToRender;

    mat4 proj = CamUBO[frame].proj;
    mat4 view = CamUBO[frame].view;
    mat4 model = mat4(inModel1, inModel2, inModel3, inModel4);

    gl_Position = getPos(proj, view, model, inPosition);

    outTBN = getTBN(inTangent, model, inNormal);

    outTexCoord = inTexCoord;
}
