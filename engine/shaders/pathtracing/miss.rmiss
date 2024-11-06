#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 2, binding = 0) uniform samplerCube cubeMap;

struct Payload {
    vec3 col;
    uint rec;
};
layout(location = 0) rayPayloadInEXT Payload payload;

void main() {
    payload.col = texture(cubeMap, gl_WorldRayDirectionEXT).rgb * 0.01;
}
