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
			if (sampleCoords.x >= 0.0 && sampleCoords.x <= 1.0 && sampleCoords.y >= 0.0 && sampleCoords.y <= 1.0) {
				float currentDepth = projCoords.z;
				shadow += texture(shadowMapSamplers[lightIndex], vec3(sampleCoords.xy, currentDepth));
			}
		}
	}

	// normalize the shadow factor
	shadow /= float(kernelSize * kernelSize);
	return shadow;
}

vec3 cookTorrance(vec3 N, vec3 L, vec3 V, vec4 albedo, float metallic, float roughness) {
	float alpha = roughness * roughness;
	float alphaS = alpha * alpha;
	float specScale = metallic; // ensure that the specular term is scaled by the metallic factor

	// compute halfway vector
	vec3 H = normalize(V + L);

	// compute the geometric term
	float NdotH = max(dot(N, H), 0.0);
	float NdotV = max(dot(N, V), 0.0);
	float VdotH = max(dot(V, H), 0.0);
	float NdotL = dot(N, L);

	// geometric attenuation factor from the view dir
	float G1V = 2.0 * NdotV / (NdotV + sqrt(alphaS + (1.0 - alphaS) * (NdotV * NdotV)));

	// geometric attenuation factor from the light dir
	float G1L = 2.0 * NdotL / (NdotL + sqrt(alphaS + (1.0 - alphaS) * (NdotL * NdotL)));
	float G = G1V * G1L;

	// compute the roughness term
	float D = alphaS / (PI * pow(NdotH * NdotH * (alphaS - 1.0) + 1.0, 2.0));

	// compute the Fresnel term (schlick approximation)
	vec3 F0 = mix(vec3(0.04), albedo.rgb, metallic);  // reflectance at normal incidence
	vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

	// specular and diffuse terms
	vec3 specular = specScale * (D * G * F) / (4.0 * NdotV * NdotL);
	vec3 diffuse = (1.0 - metallic) * albedo.rgb * (1.0 / PI);

	return (diffuse + specular); // output final color
}

void getTextures() {
	bool albedoExists = (bitfield & 1) != 0;
	bool metallicRoughnessExists = (bitfield & 2) != 0;
	bool normalExists = (bitfield & 4) != 0;
	bool emissiveExists = (bitfield & 8) != 0;
	bool occlusionExists = (bitfield & 16) != 0;

	uint nextTexture = inTexIndex;
	albedo = texture(texSamplers[nextTexture], inTexCoord);
	nextTexture += albedoExists ? 1 : 0;

	if (metallicRoughnessExists) {
		metallicRoughness = texture(texSamplers[nextTexture], inTexCoord);
		nextTexture += 1;
	}
	if (normalExists) {
		normal = (texture(texSamplers[nextTexture], inTexCoord).rgb * 2.0 - 1.0) * -1.0;
		normal = normalize(TBN * normal);
		nextTexture += 1;
	}
	if (emissiveExists) {
		emissive = texture(texSamplers[nextTexture], inTexCoord).rgb;
		nextTexture += 1;
	}
	if (occlusionExists) {
		occlusion = texture(texSamplers[nextTexture], inTexCoord).r;
	}
}

vec4 calculateLighting(bool discardTranslucent, bool fillTranslucent, float occlusionFactor) {
	vec4 color = albedo * fragColor;
	if (discardTranslucent) {
		if (color.a < 0.99) discard;
	}

	vec3 accumulated = vec3(0.0);

	float roughness = metallicRoughness.g;
	float metallic = metallicRoughness.b;

	for (int i = 0; i < lights.length(); i++) { // spotlight
		if (lights[i].intensity == 0.0) { // if the light is off, continue to next iteration
			continue;
		}

		mat4 lightView = lightMatricies[i].viewMatrix;
		mat4 lightProj = lightMatricies[i].projectionMatrix;

		float innerConeRads = lights[i].innerConeAngle * (PI / 180.0f);
		float outerConeRads = lights[i].outerConeAngle * (PI / 180.0f);
		float constAttenuation = lights[i].constantAttenuation;
		float linAttenuation = lights[i].linearAttenuation;
		float quadAttenuation = lights[i].quadraticAttenuation;

		// convert light struct to vec3s to use them in calculations
		vec3 lightPos = vec3(lights[i].pos.x, lights[i].pos.y, lights[i].pos.z); // in world space
		vec3 targetVec = vec3(lights[i].targetVec.x, lights[i].targetVec.y, lights[i].targetVec.z);
		vec3 spotDirection = normalize(lightPos - targetVec);
		vec3 fragToLightDir = normalize(lightPos - inFragPos);
		float theta = dot(spotDirection, fragToLightDir);
		vec3 lightColor = vec3(lights[i].color.x, lights[i].color.y, lights[i].color.z);

		// spotlight cutoff
		if (theta <= cos(outerConeRads)) { // if the fragment is not in the cone, continue to next iteration
			continue;
		}
		if (theta > cos(outerConeRads)) { // if inside the cone, calculate lighting
			// attenuation calculation
			float lightDistance = length(lightPos - inFragPos);
			float attenuation = 1.0 / (constAttenuation + linAttenuation * lightDistance + quadAttenuation * (lightDistance * lightDistance));
			if (attenuation < 0.005) { // early out if attenuation is too small
				continue;
			}

			vec4 fragPosLightSpace = lightProj * lightView * vec4(inFragPos, 1.0);
			float shadowFactor = shadowPCF(i, fragPosLightSpace, 4, normal, fragToLightDir);

			// spotlight intensity calculation
			float intensity;
			if (theta > cos(innerConeRads)) {
				intensity = 1.0;
			}
			else {
				intensity = (theta - cos(outerConeRads)) / (cos(innerConeRads) - cos(outerConeRads));
			}
			intensity = clamp(intensity * attenuation, 0.0, 1.0);

			// cook-torrance specular lighting
			vec3 brdf = cookTorrance(normal, fragToLightDir, inViewDir, color, metallic, roughness);
			accumulated += (lightColor * brdf * intensity) * shadowFactor;

		}
	}

	// final color calculation
	vec3 ambient = vec3(occlusionFactor) * occlusion;
	return vec4(accumulated + emissive + ambient, (fillTranslucent ? 1.0f : color.a));
}


