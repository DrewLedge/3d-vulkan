#version 460

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

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 hit;


// barycentric interpolation for vec2 - vec4
#define BARYCENTRIC(type) type barycentric##type(type b1, type b2, type b3, float u, float v) { \
    float w = 1.0 - u - v; \
    return (b1 * w) + (b2 * u) + (b3 * v); \
}

BARYCENTRIC(vec2)
BARYCENTRIC(vec3)
BARYCENTRIC(vec4)

mat3 getTBN(vec4 tangent, vec3 normal) {
    float handedness = tangent.w;

    mat3 normMat = transpose(inverse(mat3(gl_ObjectToWorldEXT)));
    vec3 N = normalize(normMat * normal);
    vec3 T = normalize(normMat * tangent.xyz);
    T = normalize(T - dot(T, N) * N); // re orthogonalize tangent
    vec3 B = normalize(cross(N, T) * handedness);
    return mat3(T, B, N);
}

vec4 albedo = vec4(0.0f);
vec4 metallicRoughness = vec4(1.0f);
vec3 normal = vec3(1.0f);
vec3 emissive = vec3(0.0f);
float occlusion = 1.0f;

void getTextures(uint bitfield, uint texIndex, vec2 uv, mat3 tbn) {
    bool albedoExists = (bitfield & 1) != 0;
    bool metallicRoughnessExists = (bitfield & 2) != 0;
    bool normalExists = (bitfield & 4) != 0;
    bool emissiveExists = (bitfield & 8) != 0;
    bool occlusionExists = (bitfield & 16) != 0;

    uint nextTexture = texIndex;
    albedo = texture(texSamplers[nextTexture], uv);
    nextTexture += albedoExists ? 1 : 0;

    if (metallicRoughnessExists) {
    	metallicRoughness = texture(texSamplers[nextTexture], uv);
    	nextTexture += 1;
    }
    if (normalExists) {
    	normal = (texture(texSamplers[nextTexture], uv).rgb * 2.0 - 1.0);
    	normal.y *= -1.0;
    	normal = normalize(tbn * normal);
    	nextTexture += 1;
    }
    if (emissiveExists) {
    	emissive = texture(texSamplers[nextTexture], uv).rgb;
    	nextTexture += 1;
    }
    if (occlusionExists) {
    	occlusion = texture(texSamplers[nextTexture], uv).r;
    }
}

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

void main() {
    uint index = 3 * gl_PrimitiveID;

    vec2 uv;
    vec4 color;
    vec3 norm;
    vec4 tangent;

    getVertData(index, uv, color, norm, tangent);
    
    albedo *= color;

    mat3 tbn = getTBN(tangent, norm);

    uint texindex = texIndices[gl_InstanceCustomIndexEXT].albedo;
    uint bitfield = texIndices[gl_InstanceCustomIndexEXT].bitfield;

    getTextures(bitfield, texindex, uv, tbn);

    payload += emissive;
}