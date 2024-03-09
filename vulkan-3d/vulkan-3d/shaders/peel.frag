#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 0) uniform sampler2D texSamplers[];
layout(input_attachment_index = 0, set = 2, binding = 5) uniform subpassInput prevDepth;


layout(location = 0) in vec4 inColor;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec4 outFragColor;
layout(location = 1) out float outDepth;

layout(push_constant) uniform PC {
    int albedo; // the index of the base texture in texSamplers
    int peelInd; // the current peel index
    int numPeels;
} pc;

void main() {
    vec4 albedoColor = texture(texSamplers[pc.albedo], inTexCoord);
    float depth = gl_FragCoord.z;

    float alpha = albedoColor.a * inColor.a;
    if (alpha == 0.0f || alpha > 0.99f) discard;
    
    if (pc.peelInd == 0) { // check if this is the first peel
        // output the color and depth for the first peel
        outFragColor = albedoColor;
        outDepth = depth;
    } else {
        float prevDepthVal = subpassLoad(prevDepth).r; // load the depth value from the previous peel
        
        // check if the current depth is greater than the previous depth
        if (depth > prevDepthVal) {
            // output the color and depth for the current peel
            outFragColor = albedoColor;
            outDepth = depth;
        } else {
            discard; // discard if its hidden by the previous peel
        }
    }
}

