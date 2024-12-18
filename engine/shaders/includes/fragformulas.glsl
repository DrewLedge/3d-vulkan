#define PI 3.141592653589793238f


// calc the geometry function for a given term using Schlick-GGX approximation
// the geometry function accounts for the fact that some microfacets may be shadowed by others, which reduces the reflectance
// without the geometry function, rough surfaces would appear overly shiny
float gSchlickGGX(float term, float k) {
    return term / (term * (1.0f - k) + k);
}

// calc the geometry function based on the light and view dir
// this determines which microfacets are shadowed, and thus cannot reflect light into the view dir
float gSmith(float NdotV, float NdotL, float roughness) {
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;
    return gSchlickGGX(NdotV, k) * gSchlickGGX(NdotL, k);
}

// calc the normal distribution function (ndf) using Trowbridge-Reitz model
// the ndf determines the statistical distribution of microfacet orientations that contribute to the reflection
// ndf is crucial for determining the intensity and shape of specular highlights
float ndf(float NdotH, float a) {
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom);
}

// calc the fresnel term using Schlick approximation
// the fresnel term determines how reflective the material is based on the viewing angle and metallic value
vec3 fresnelTerm(vec3 color, float metallic, float VdotH) {
    const vec3 br = vec3(0.04f); // the base reflectivity constant for non metallic (dielectric) materials
    vec3 F0 = mix(br, color, metallic); // the base reflectivity of the material based on the metallic value and the albedo
    return F0 + (1.0f - F0) * pow(1.0f - VdotH, 5.0f);
}

vec3 cookTorrance(vec3 N, vec3 L, vec3 V, vec4 albedo, float metallic, float roughness) {
    float a = roughness * roughness;

    // compute halfway vector
    vec3 H = normalize(V + L);

    // compute the dot products
    float NdotH = max(dot(N, H), 0.0f);
    float NdotV = max(dot(N, V), 0.0f);
    float VdotH = max(dot(V, H), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);

    // normal distribution function
    float ND = ndf(NdotH, a);

    // geometry function
    float G = gSmith(NdotV, NdotL, roughness);

    // fresnel term
    vec3 F = fresnelTerm(albedo.rgb, metallic, VdotH);

    float norm = (4.0f * max(NdotV * NdotL, 0.0001f)); // used to normalize the specular term
    vec3 spec = (ND * G * F) / norm;

    // the proportion of light not reflected specularly
    vec3 kD = vec3(1.0f) - F;
    kD *= 1.0f - metallic;

    vec3 diffuse = kD * albedo.rgb / PI;

    return (diffuse + spec) * NdotL;
}

void getTextures(uint bitfield, uint texIndex, vec2 uv, mat3 tbn, out vec4 albedo, out vec4 metallicRoughness, out vec3 normal, out vec3 emissive, out float occlusion) {
    bool albedoExists = (bitfield & 1) != 0;
    bool metallicRoughnessExists = (bitfield & 2) != 0;
    bool normalExists = (bitfield & 4) != 0;
    bool emissiveExists = (bitfield & 8) != 0;
    bool occlusionExists = (bitfield & 16) != 0;

    // default values
    albedo = vec4(1.0f);
    metallicRoughness = vec4(0.0f, 0.5f, 0.0f, 1.0f);
    normal = vec3(0.0f);
    emissive = vec3(0.0f);
    occlusion = 1.0f;

    uint nextTexture = texIndex;
    albedo = texture(texSamplers[nextTexture], uv);
    nextTexture += albedoExists ? 1 : 0;

    if (metallicRoughnessExists) {
        metallicRoughness = texture(texSamplers[nextTexture], uv);
        nextTexture += 1;
    }
    if (normalExists) {
        normal = (texture(texSamplers[nextTexture], uv).rgb * 2.0f - 1.0f);
        normal = normalize(tbn * normal);
        nextTexture += 1;
    }
    if (emissiveExists) {
        emissive = texture(texSamplers[nextTexture], uv).rgb;
        nextTexture += 1;
    }
    if (occlusionExists) {
        occlusion = texture(texSamplers[nextTexture], uv).r * metallicRoughness.b;
    }
}

float linDepth(float depth, float near, float far) {
    float n = far * near;
    float d = far + depth * (near - far);

    return n / d;
}

vec2 getTexCords(sampler2D tex, vec2 fragCoord) {
    ivec2 texDimensions = textureSize(tex, 0);
    return fragCoord / texDimensions;
}

float calcFallofff(float outer, float inner, float theta) {
    float f = smoothstep(cos(outer), cos(inner), theta);
    return f * f;
}

vec3 getFragPos(vec2 uv, float depth, mat4 iproj, mat4 iview) {
    // convert uv to -1 to 1 range
    vec2 ndc = uv * 2.0f - 1.0f;

    // get the clip space pos
    vec4 clip = vec4(ndc, depth, 1.0f);

    // multiply by the inverse proj mat to get the pos in view space
    vec4 view = iproj * clip;
    view /= view.w; // perspective divide

    // multiply by the inverse view mat to get the pos in world space
    return (iview * view).xyz;
}

vec3 getViewDir(vec3 fragWorldPos, mat4 iview) {
    vec3 camPos = vec3(iview[3]);
    return normalize(camPos - fragWorldPos);
}
