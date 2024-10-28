#version 460

#define PI 3.141592653589793238

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require

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

    uint64_t vertexAddress;
    uint64_t indexAddress;
};

layout (set = 6, binding = 0) readonly buffer TexIndexBuffer {
    TexIndexData texIndices[];
};

struct Vertex {
    vec3 pos;
    vec2 tex;
    vec4 col;
    vec3 normal;
    vec4 tangent;
};

layout(buffer_reference) readonly buffer VertBuffer {
    Vertex vertices[];
};

layout(buffer_reference) readonly buffer IndexBuffer { 
    uint indices[];
};

struct Payload {
    vec3 col;
    uint rec;
};

layout(location = 0) rayPayloadInEXT Payload payload;
layout(location = 1) rayPayloadEXT bool shadowPayload;
hitAttributeEXT vec2 hit;


// barycentric interpolation for vec2 - vec4
#define BARYCENTRIC(type) type barycentric##type(type b1, type b2, type b3, float u, float v) { \
    float w = 1.0 - u - v; \
    return (b1 * w) + (b2 * u) + (b3 * v); \
}

BARYCENTRIC(vec2)
BARYCENTRIC(vec3)
BARYCENTRIC(vec4)

vec4 albedo = vec4(0.0f);
vec4 metallicRoughness = vec4(1.0f);
vec3 normal = vec3(1.0f);
vec3 emissive = vec3(0.0f);
float occlusion = 1.0f;

void getVertData(uint index, out vec2 uv, out vec4 color, out vec3 normal, out vec4 tangent) {
    uint64_t vertAddr = texIndices[gl_InstanceCustomIndexEXT].vertexAddress;
    uint64_t indexAddr = texIndices[gl_InstanceCustomIndexEXT].indexAddress;

    IndexBuffer indexBuffer = IndexBuffer(indexAddr);
    VertBuffer vertBuffer = VertBuffer(vertAddr);

    uint i1 = indexBuffer.indices[index + 0];
    uint i2 = indexBuffer.indices[index + 1];
    uint i3 = indexBuffer.indices[index + 2];

    uint[3] indices = uint[3](i1, i2, i3);
    vec2[3] uvs;
    vec4[3] cols;
    vec3[3] normals; 
    vec4[3] tangents;

    for (uint i = 0; i < 3; i++) {
        uvs[i] = vertBuffer.vertices[indices[i]].tex;
        cols[i] = vertBuffer.vertices[indices[i]].col;
        normals[i] = vertBuffer.vertices[indices[i]].normal;
        tangents[i] = vertBuffer.vertices[indices[i]].tangent;
    }

    float u = hit.x;
    float v = hit.y;

    uv = barycentricvec2(uvs[0], uvs[1], uvs[2], u, v);
    color = barycentricvec4(cols[0], cols[1], cols[2], u, v);
    normal = barycentricvec3(normals[0], normals[1], normals[2], u, v);
    tangent = barycentricvec4(tangents[0], tangents[1], tangents[2], u, v);
}

#include "../includes/vertformulas.glsl"
#include "../includes/fragformulas.glsl"

void main() {
    if (payload.rec >= 4) {
        payload.col = vec3(0);
        return;
    }

    uint index = 3 * gl_PrimitiveID;

    vec2 uv;
    vec4 color;
    vec3 norm;
    vec4 tangent;

    getVertData(index, uv, color, norm, tangent);
    
    albedo *= color;

    mat3 tbn = getTBN(tangent, mat3(gl_ObjectToWorldEXT), norm);

    uint texindex = texIndices[gl_InstanceCustomIndexEXT].albedo;
    uint bitfield = texIndices[gl_InstanceCustomIndexEXT].bitfield;

    getTextures(bitfield, texindex, uv, tbn);

    vec3 hitPos = gl_WorldRayOriginEXT + (gl_WorldRayDirectionEXT * gl_HitTEXT);
    vec3 viewDir = -gl_WorldRayDirectionEXT;
    vec3 reflectDir = reflect(gl_WorldRayDirectionEXT, normal);

    float roughness = metallicRoughness.g;
    float metallic = metallicRoughness.b;

    vec3 accumulated = vec3(0.0);

    payload.rec++;

    traceRayEXT(
        TLAS,
        gl_RayFlagsOpaqueEXT,
        0xFF,                 // cull mask
        0,                    // sbt offset
        0,                    // sbt stride
        0,                    // miss index
        hitPos,               // pos
        0.01,                 // min-range
        reflectDir,           // dir
        100.0,                // max-range
        0                     // payload
    );

    payload.rec--;

    // fresnel term
    vec3 H = normalize(viewDir + reflectDir);
    float VdotH = max(dot(viewDir, H), 0.0);
    vec3 F = fresnelTerm(albedo.rgb, metallic, VdotH);

    float r = pow(1.0 - roughness, 4);

    // get the reflection color
    vec3 refl = payload.col * F * r;

    for (int i = 0; i < lights.length(); i++) { // spotlight
        if (lights[i].intensity == 0.0) continue;

        float innerConeRads = radians(lights[i].innerConeAngle);
        float outerConeRads = radians(lights[i].outerConeAngle);
        float constAttenuation = lights[i].constantAttenuation;
        float linAttenuation = lights[i].linearAttenuation;
        float quadAttenuation = lights[i].quadraticAttenuation;

        // convert light struct to vec3s to use them in calculations
        vec3 lightPos = lights[i].pos.xyz;
        vec3 targetVec = lights[i].targetVec.xyz;
        vec3 lightColor = lights[i].color.xyz;

        vec3 spotDirection = normalize(lightPos - targetVec);
        vec3 fragToLightDir = normalize(lightPos - hitPos);
        float theta = dot(spotDirection, fragToLightDir);

        // spotlight cutoff
        if (theta > cos(outerConeRads)) { // if inside the cone, calculate lighting
            float blend = smoothstep(cos(outerConeRads), cos(innerConeRads), theta);

            // attenuation calculation
            float lightDistance = distance(lightPos, hitPos);
            float attenuation = 1.0 / (constAttenuation + linAttenuation * lightDistance + quadAttenuation * (lightDistance * lightDistance));
            if (attenuation < 0.01) continue;

            // get the contribution
            float contribution = lights[i].intensity * blend * attenuation;
            if (contribution < 0.01) continue;

            float min = 0.001;
            float max = lightDistance - min;

            // trace the shadow rays
            traceRayEXT(
                TLAS,
                gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                0xFF,                 // cull mask
                1,                    // sbt offset
                0,                    // sbt stride
                1,                    // miss index
                hitPos,               // pos
                min,                  // min-range
                fragToLightDir,       // dir
                max,                  // max-range
                1                     // payload
            );
            
            if (shadowPayload) continue;

            vec3 brdf = cookTorrance(normal, fragToLightDir, viewDir, albedo, metallic, roughness);
            vec3 direct = lightColor * brdf;

            // add the direct lighting with the reflection color
            accumulated += (direct + refl) * contribution;
            }
    }

    // final color calculation
    vec3 ambient = vec3(0.005) * occlusion;
    payload.col = accumulated + emissive + ambient;
}