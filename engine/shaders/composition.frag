#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
    // get the color and depth from the main pass 
    vec4 mainColor = texture(textures[0], inUV);

    // get the weighted color and alpha from the wboit pass
    vec4 weightedColor = texture(textures[1], inUV);
    float weightedAlpha = weightedColor.a;

    // if there is no weighted color, early out
    if (weightedColor == vec4(0.0, 0.0, 0.0, 1.0)) {
        outColor = mainColor;
        return;
    }

    weightedColor.rgb /= max(weightedAlpha, 1e-5);

    // blend the main color with the weighted color based on the weighted alpha
    outColor = vec4(mix(mainColor.rgb, weightedColor.rgb, weightedAlpha), 1.0);
    outColor.rgb *= pow(weightedAlpha, 0.4);
}
