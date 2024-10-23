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

layout (set = 1, binding = 0) readonly buffer LightBuffer {
    LightData lights[];
};

layout(set = 4, binding = 0) uniform accelerationStructureEXT TLAS;

layout(location = 0) rayPayloadInEXT vec4 payload;

void main() {
    payload = vec4(1.0, 0.0, 0.0, 1.0); // red
}