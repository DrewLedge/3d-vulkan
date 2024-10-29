#define PI 3.141592653589793238

// outputs a float based on the self shadowing of microfacets
// it accounts for the fact that some microfacets may be shadowed by others, which reduces the reflectance
// without the geometric attenuation, rough surfaces would appear overly shiny
float gAttenutation(float term, float alpha) {
    return 2.0 * term / (term + sqrt(alpha + (1.0 - alpha) * (term * term)));
}

// outputs a float based on the statistical distribution of microfacet orientations
// essentially, it determins how likely a microfacet is oriented in which it will reflect light
// ndf is crucial for determining the intensity and shape of specular highlights
float ndf(float NdotH, float alpha) {
    return alpha / (PI * pow(NdotH * NdotH * (alpha - 1.0) + 1.0, 2.0));
}

// outputs a vec3 for to simulate how the reflectivity changes based on the viewing angle
vec3 fresnelTerm(vec3 color, float metallic, float VdotH) {
    // F0 represents the base reflectivity
    // which is the amount of light the material reflects from straight on
    // used to distinguish between metallic things and non metallic things (dielectrics)
    vec3 F0 = mix(vec3(0.04), color, metallic);
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0); // uses schlick approximation to get the fresnel term
}

vec3 cookTorrance(vec3 N, vec3 L, vec3 V, vec4 albedo, float metallic, float roughness) {
    float alpha = roughness * roughness;

    // compute halfway vector
    vec3 H = normalize(V + L);

    // compute the dot products
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float NdotL = max(dot(N, L), 0.0);

    // normal distribution function
    float ND = ndf(NdotH, alpha);

    // geometric attenuation factor
    float G = gAttenutation(NdotV, roughness) * gAttenutation(NdotL, roughness);

    // fresnel term
    vec3 F = fresnelTerm(albedo.rgb, metallic, VdotH);

    float norm = (4.0 * max(NdotV * NdotL, 0.0001)); // used to normalize the specular term

    // specular and diffuse components
    vec3 specular = (ND * G * F) / norm;
    vec3 diffuse = (1.0 - metallic) * albedo.rgb / PI;

    return (diffuse + specular);
}

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

float linDepth(float depth, float near, float far) {
    float z = depth * 2.0 - 1.0; // to ndc
    return (2.0 * near * far) / (far + near - z * (far - near));
}

vec2 getTexCords(sampler2D tex, vec2 fragCoord) {
    ivec2 texDimensions = textureSize(tex, 0);
    return fragCoord / texDimensions;
}
