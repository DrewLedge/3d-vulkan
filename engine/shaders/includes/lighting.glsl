#define PI 3.141592653589793238

// get the PCF shadow factor (used for softer shadows)
float shadowPCF(int lightIndex, vec4 fragPosLightSpace, int kernelSize, vec3 norm, vec3 lightDir) {
    int halfSize = kernelSize / 2;
    float shadow = 0.0;
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    // calculate texel size based on shadow map dimensions
    vec2 texelSize = 1.0 / textureSize(shadowMapSamplers[lightIndex], 0);

    // loop through the PCF kernel
    for (int x = -halfSize; x <= halfSize; ++x) {
        for (int y = -halfSize; y <= halfSize; ++y) {
            // sample the depth from shadow map
            vec2 sampleCoords = projCoords.xy + vec2(x, y) * texelSize;
            float currentDepth = projCoords.z;
            shadow += texture(shadowMapSamplers[lightIndex], vec3(sampleCoords.xy, currentDepth));
        }
    }

    // normalize the shadow factor
    shadow /= float(kernelSize * kernelSize);
    return shadow;
}


vec4 calcLighting(bool discardTranslucent, bool discardOpaque) {
    if (discardTranslucent && albedo.a < 0.95) discard;
    if (discardOpaque && albedo.a >= 0.95) discard;

    vec3 accumulated = vec3(0.0);

    float roughness = metallicRoughness.g;
    float metallic = metallicRoughness.b;

    for (int i = 0; i < lights.length(); i++) {
        if (lights[i].intensity < 0.01) continue;

        float inner = lights[i].innerConeAngle;
        float outer = lights[i].outerConeAngle;
        float constAttenuation = lights[i].constantAttenuation;
        float linAttenuation = lights[i].linearAttenuation;
        float quadAttenuation = lights[i].quadraticAttenuation;

        vec3 lightPos = lights[i].pos.xyz;
        vec3 target = lights[i].target.xyz;
        vec3 lightColor = lights[i].color.xyz;

        vec3 spotDir = normalize(lightPos - target);
        vec3 fragToLightDir = normalize(lightPos - inFragPos);
        float theta = dot(spotDir, fragToLightDir);

        // if the fragment is outside the cone, early out
        if (theta <= cos(outer)) continue;

        // shadow factor
        vec4 fragPosLightSpace = lights[i].proj * lights[i].view * vec4(inFragPos, 1.0);
        float shadowFactor = shadowPCF(i, fragPosLightSpace, 4, normal, fragToLightDir);
        if (shadowFactor < 0.01) continue;

        // attenuation
        float lightDistance = distance(lightPos, inFragPos);
        float attenuation = 1.0 / (constAttenuation + linAttenuation * lightDistance + quadAttenuation * (lightDistance * lightDistance));
        if (attenuation < 0.01) continue;

        // get the contribution
        float contribution = lights[i].intensity * attenuation * calcFallofff(outer, inner, theta);
        if (contribution < 0.01) continue;

        vec3 brdf = cookTorrance(normal, fragToLightDir, inViewDir, albedo, metallic, roughness);
        accumulated += (brdf * lightColor * contribution * shadowFactor);
    }

    // final color calculation
    vec3 o = albedo.rgb * occlusion * 0.005;
    return vec4(accumulated + emissive + o, albedo.a);
}
