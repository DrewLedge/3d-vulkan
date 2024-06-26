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

float gAttenutation(float term, float alphaS) {
	return 2.0 * term / (term + sqrt(alphaS + (1.0 - alphaS) * (term * term)));
}

float roughnessTerm(float NdotH, float alphaS) {
	return alphaS / (PI * pow(NdotH * NdotH * (alphaS - 1.0) + 1.0, 2.0));
}

vec3 frenselTerm(vec3 color, float metallic, float VdotH) {
	vec3 F0 = mix(vec3(0.04), color, metallic);  // base reflectivity
	return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

vec3 cookTorrance(vec3 N, vec3 L, vec3 V, vec4 albedo, float metallic, float roughness) {
	float alpha2 = roughness * roughness;

	// compute halfway vector
	vec3 H = normalize(V + L);

	// compute the dot products
	float NdotH = max(dot(N, H), 0.0);
	float NdotV = max(dot(N, V), 0.0);
	float VdotH = max(dot(V, H), 0.0);
	float NdotL = max(dot(N, L), 0.0);

	// compute the roughness term
	float D = roughnessTerm(NdotH, alpha2);

	// geometric attenuation factor
	float G = gAttenutation(NdotV, roughness) * gAttenutation(NdotL, roughness);

	// compute the Fresnel term (schlick approximation)
	vec3 F = frenselTerm(albedo.rgb, metallic, VdotH);

	// specular and diffuse terms
	vec3 specular = (D * G * F) / (4.0 * max(NdotV * NdotL, 0.0001));
	vec3 diffuse = (1.0 - metallic) * albedo.rgb / PI;

	return (diffuse + specular); // output final color
}

void getTextures() {
	bool albedoExists = (inBitfield & 1) != 0;
	bool metallicRoughnessExists = (inBitfield & 2) != 0;
	bool normalExists = (inBitfield & 4) != 0;
	bool emissiveExists = (inBitfield & 8) != 0;
	bool occlusionExists = (inBitfield & 16) != 0;

	uint nextTexture = inTexIndex;
	albedo = texture(texSamplers[nextTexture], inTexCoord);
	nextTexture += albedoExists ? 1 : 0;

	if (metallicRoughnessExists) {
		metallicRoughness = texture(texSamplers[nextTexture], inTexCoord);
		nextTexture += 1;
	}
	if (normalExists) {
		normal = (texture(texSamplers[nextTexture], inTexCoord).rgb * 2.0 - 1.0);
		normal.y *= -1.0;
		normal = normalize(inTBN * normal);
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

vec4 calcLighting(bool discardTranslucent, bool discardOpaque, float occlusionFactor) {
	vec4 color = albedo * inFragColor;
	if (discardTranslucent && color.a < 0.98) discard;
	if (discardOpaque && color.a >= 0.98) discard;

	vec3 accumulated = vec3(0.0);

	float roughness = metallicRoughness.g;
	float metallic = metallicRoughness.b;

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
		vec3 fragToLightDir = normalize(lightPos - inFragPos);
		float theta = dot(spotDirection, fragToLightDir);

		// spotlight cutoff
		if (theta > cos(outerConeRads)) { // if inside the cone, calculate lighting
			float blend = smoothstep(cos(outerConeRads), cos(innerConeRads), theta);

			// attenuation calculation
			float lightDistance = distance(lightPos, inFragPos);
			float attenuation = 1.0 / (constAttenuation + linAttenuation * lightDistance + quadAttenuation * (lightDistance * lightDistance));
			if (attenuation < 0.01) continue;

			// get the shadow factor
			vec4 fragPosLightSpace = lights[i].proj * lights[i].view * vec4(inFragPos, 1.0);
			float shadowFactor = shadowPCF(i, fragPosLightSpace, 4, normal, fragToLightDir);
			if (shadowFactor < 0.01) continue;

			// get the intensity
			float intensity = lights[i].intensity * blend * attenuation;
			if (intensity < 0.01) continue;

			// cook-torrance specular lighting
			vec3 brdf = cookTorrance(normal, fragToLightDir, inViewDir, color, metallic, roughness);
			accumulated += (lightColor * brdf * intensity * blend) * shadowFactor;
		}
	}

	// final color calculation
	vec3 ambient = vec3(occlusionFactor) * occlusion;
	return vec4(accumulated + emissive + ambient, color.a);
}

float linDepth(float depth, float near, float far) {
	float z = depth * 2.0 - 1.0; // to ndc
	return (2.0 * near * far) / (far + near - z * (far - near));
}

vec2 getTexCords(sampler2D tex) {
	ivec2 texDimensions = textureSize(tex, 0);
	return gl_FragCoord.xy / texDimensions;
}


