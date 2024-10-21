#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform sampler2D texSamplers[];

struct LightData {
    mat4 view;
    mat4 proj;

    vec4 pos;
    vec4 color;
    vec4 targetVec;
    float intensity;
	float innerConeAngle; // in degrees
	float outerConeAngle; // in degrees
	float constantAttenuation;
	float linearAttenuation;
	float quadraticAttenuation;
};

layout (set = 1, binding = 1) readonly buffer LightBuffer {
    LightData lights[];
};

layout(set = 2, binding = 7) uniform accelerationStructureEXT TLAS;

void main() {
}