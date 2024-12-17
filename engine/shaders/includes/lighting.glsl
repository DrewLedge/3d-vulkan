#define PI 3.141592653589793238f

// get the PCF shadow factor (used for softer shadows)
float shadowPCF(int frame, int frameCount, int lightIndex, vec4 fragPosLightspace, int kernelSize, vec3 norm, vec3 lightDir) {
    int halfSize = kernelSize / 2;
    float shadow = 0.0f;
    vec3 projCoords = fragPosLightspace.xyz / fragPosLightspace.w;
    projCoords.xy = projCoords.xy * 0.5f + 0.5f;

    // calculate texel size based on shadow map dimensions
    int shadowTexIndex = (lightIndex * frameCount) + frame;
    vec2 texelSize = 1.0f / textureSize(shadowMapSamplers[shadowTexIndex], 0);

    // loop through the PCF kernel
    for (int x = -halfSize; x <= halfSize; ++x) {
        for (int y = -halfSize; y <= halfSize; ++y) {
            // sample the depth from shadow map
            vec2 sampleCoords = projCoords.xy + vec2(x, y) * texelSize;
            float currentDepth = projCoords.z;
            shadow += texture(shadowMapSamplers[shadowTexIndex], vec3(sampleCoords.xy, currentDepth));
        }
    }

    // normalize the shadow factor
    shadow /= float(kernelSize * kernelSize);
    return shadow;
}

vec4 calcLighting(vec4 albedo, vec4 metallicRoughness, vec3 normal, vec3 emissive, float occlusion, vec3 fragPos, vec3 viewDir, int frame, int frameCount, int lightCount, bool discardTranslucent, bool discardOpaque) {
    if (discardTranslucent && albedo.a < 0.95f) discard;
    if (discardOpaque && albedo.a >= 0.95f) discard;

    vec3 accumulated = vec3(0.0f);

    float roughness = metallicRoughness.g;
    float metallic = metallicRoughness.b;

    for (int i = 0; i < lightCount; i++) {
        if (lssbo[frame].lights[i].intensity < 0.01f) continue;

        float inner = lssbo[frame].lights[i].innerConeAngle;
        float outer = lssbo[frame].lights[i].outerConeAngle;
        float constAttenuation = lssbo[frame].lights[i].constantAttenuation;
        float linAttenuation = lssbo[frame].lights[i].linearAttenuation;
        float quadAttenuation = lssbo[frame].lights[i].quadraticAttenuation;

        vec3 lightPos = lssbo[frame].lights[i].pos.xyz;
        vec3 target = lssbo[frame].lights[i].target.xyz;
        vec3 lightColor = lssbo[frame].lights[i].color.xyz;

        vec3 spotDir = normalize(lightPos - target);
        vec3 fragToLightDir = normalize(lightPos - fragPos);
        float theta = dot(spotDir, fragToLightDir);

        // if the fragment is outside the cone, early out
        if (theta <= cos(outer)) continue;

        // shadow factor
        vec4 fragPosLightspace = lssbo[frame].lights[i].proj * lssbo[frame].lights[i].view * vec4(fragPos, 1.0f);
        float shadowFactor = shadowPCF(frame, frameCount, i, fragPosLightspace, 2, normal, fragToLightDir);
        if (shadowFactor < 0.01f) continue;

        // attenuation
        float lightDistance = distance(lightPos, fragPos);
        float attenuation = 1.0f / (constAttenuation + linAttenuation * lightDistance + quadAttenuation * (lightDistance * lightDistance));
        if (attenuation < 0.01f) continue;

        // get the contribution
        float contribution = lssbo[frame].lights[i].intensity * attenuation * calcFallofff(outer, inner, theta);
        if (contribution < 0.01f) continue;

        vec3 brdf = cookTorrance(normal, fragToLightDir, viewDir, albedo, metallic, roughness);
        accumulated += (brdf * lightColor * contribution * shadowFactor);
    }

    // final color calculation
    vec3 o = albedo.rgb * occlusion * 0.005f;
    return vec4(accumulated + emissive + o, albedo.a);
}
