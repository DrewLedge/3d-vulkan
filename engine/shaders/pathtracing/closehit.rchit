#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

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

struct TexIndexData {
    uint albedo;
    uint bitfield;
};

layout (set = 6, binding = 0) readonly buffer TexIndexBuffer {
    TexIndexData texIndices[];
};

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    uint index = texIndices[gl_InstanceCustomIndexEXT].albedo;
    float f = 1.0 / index;
    payload += vec3(f);
   
}