#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 6) uniform sampler2D depthPeels[];
layout(set = 1, binding = 7) uniform sampler2D mainPass[];

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    int numPeels;
} pc;

void main() {
    // get the color and depth from the main pass 
    vec4 mainColor = texture(mainPass[0], inUV);
    float mainDepth = texture(mainPass[1], inUV).r;

    vec4 finalColor = vec4(0.0);
    float finalAlpha = 1.0;
    float finalDepth = 0.0;

    // initialize the final color with the main pass color
    finalColor = mainColor;
    finalDepth = mainDepth;

    // blend the depth peels with the main pass
    for (int i = 0; i < pc.numPeels; i++) {
        // color and depth of the current peel
        vec4 peelColor = texture(depthPeels[i], inUV);
        float peelDepth = texture(depthPeels[i + pc.numPeels], inUV).r;

        if (peelDepth < finalDepth) {
            float peelAlpha = 1.0 - peelColor.a; // get the inverse alpha

            // blend the peel color with the final color
            finalColor += peelColor * finalAlpha;
            finalAlpha *= peelAlpha;

            // update the final depth with the current peel depth
            finalDepth = peelDepth;
        }
    }
    outColor = finalColor;
}

