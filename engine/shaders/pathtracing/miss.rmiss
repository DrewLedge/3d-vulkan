#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 2, binding = 0) uniform samplerCube cubeMap;

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
	payload = texture(cubeMap, gl_WorldRayDirectionEXT).rgb;
}