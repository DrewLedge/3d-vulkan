#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 0, binding = 5) uniform sampler2D textures[];

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
    // get the color and depth from the main pass 
    vec4 mainColor = texture(textures[0], inUV);
    float mainDepth = texture(textures[1], inUV).r;

    // get the weighted color and alpha from the wboit pass
    vec4 weightedColor = texture(textures[2], inUV);
    float weightedAlpha = texture(textures[3], inUV).r;

    float wboitDepth = texture(textures[4], inUV).r;

    if (wboitDepth >= mainDepth) {
        outColor = mainColor;
        return;
    }

    vec3 blended = weightedColor.rgb / max(weightedAlpha, 0.00001); // blend the weighted color and alpha

    // mix the main color with the blended color using the weighted alpha
    outColor = vec4(mix(mainColor.rgb, blended, (1.0f - weightedAlpha)), 1.0);
}
