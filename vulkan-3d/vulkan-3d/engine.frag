#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 1, binding = 1) uniform sampler2D texSamplers[];

layout(set = 4, binding = 4) uniform sampler2D  shadowMapSamplers[];

struct formsVec3 { // custom structure to hold my vec3s
    float x;
    float y;
    float z;
};
struct lightMatrix {
    mat4 viewMatrix;
    mat4 projectionMatrix;
};
struct lightData {
    formsVec3 pos;
    formsVec3 color;
    formsVec3 targetVec;
    float intensity;
	float innerConeAngle; // in degrees
	float outerConeAngle; // in degrees
	float constantAttenuation;
	float linearAttenuation;
	float quadraticAttenuation;
};


layout (set=3, binding = 3) buffer LightBuffer {
	lightMatrix lightMatricies[20];
    lightData lights[20];
};

layout(location = 0) in vec4 fragColor;
layout(location = 2) in float inAlpha;
layout(location = 3) in vec2 inTexCoord;
layout(location = 5) flat in uint inTexIndex;
layout(location = 6) flat in uint inModelIndex;
layout(location = 7) in vec3 inFragPos;
layout(location = 8) in vec3 inNormal;
layout(location = 9) in vec3 inViewDir;
layout(location = 10) in vec3 inCamPos;
layout(location = 0) out vec4 outColor;
float PI = acos(-1.0);

float cookTorranceSpec(vec3 normal, vec3 lightDir, vec3 viewDir, float roughness, vec3 f0) { // fix
    vec3 halfVector = normalize(lightDir + viewDir); // half vector between light and view direction
    float NdotH = max(dot(normal, halfVector), 0.0); // dot product between normal and half vector
    float NdotV = max(dot(normal, viewDir), 0.0); // dot product between normal and view direction
    float VdotH = max(dot(viewDir, halfVector), 0.0); // dot product between view direction and half vector

    // geometric attenuation
    float geometric = min(1.0, min((2.0 * NdotH * NdotV) / VdotH, (2.0 * NdotH * NdotV) / NdotH));
    float denominator = PI * pow(roughness, 2) * pow(NdotH, 4); // denominator of the equation
    return geometric / denominator; // return specular intensity
}


// get the PCF shadow factor (used for softer shadows)
float shadowPCF(int lightIndex, vec4 fragPosLightSpace, int kernelSize, vec3 norm, vec3 lightDir) {  
    int halfSize = kernelSize / 2;
    float shadow = 0.0;
    vec3 projCoords = fragPosLightSpace.xyz;

    // calculate texel size based on shadow map dimensions
    vec2 texelSize = 1.0 / textureSize(shadowMapSamplers[lightIndex], 0);

    // loop through the PCF kernel
    for(int x = -halfSize; x <= halfSize; ++x) {
        for(int y = -halfSize; y <= halfSize; ++y) {
            // sample the depth from shadow map
            vec2 sampleCoords = clamp(projCoords.xy + vec2(x, y) * texelSize, 0.0, 1.0);
            float pcfDepth = texture(shadowMapSamplers[lightIndex], sampleCoords).r;
            float currentDepth = projCoords.z;

            // perform depth comparison
            float bias = max(0.005 * (1.0 - dot(norm, lightDir)), 0.005);
            shadow += (currentDepth - bias > pcfDepth) ? 1.0 : 0.0;

        }
    }
    // normalize the shadow factor
    shadow /= float(kernelSize * kernelSize);

    return shadow;
}


void main() {
    vec4 albedo = texture(texSamplers[inTexIndex], inTexCoord);
    vec4 metallicRoughness = texture(texSamplers[inTexIndex + 1], inTexCoord);
    vec4 normalMap = texture(texSamplers[inTexIndex + 2], inTexCoord);

    vec3 normal = normalize(normalMap.xyz * 2.0 - 1.0);
    vec3 color = albedo.rgb;
        
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    for (int i = 0; i < lights.length(); i++) { // spotlight
         float innerConeRads = lights[i].innerConeAngle * (PI/180.0f);
         float outerConeRads = lights[i].outerConeAngle * (PI/180.0f);
         float constAttenuation = lights[i].constantAttenuation;
	     float linAttenuation = lights[i].linearAttenuation;
	     float quadAttenuation = lights[i].quadraticAttenuation;

		 // convert light struct to vec3s so I can use them in calculations
		 vec3 lightPos = vec3(lights[i].pos.x, lights[i].pos.y, lights[i].pos.z); // in world space
         vec3 targetVec = vec3(lights[i].targetVec.x, lights[i].targetVec.y, lights[i].targetVec.z);
         vec3 spotDirection = normalize(lightPos - targetVec);
         vec3 fragToLightDir = normalize(lightPos - inFragPos);
         float theta = dot(spotDirection, fragToLightDir);
         vec3 lightColor = vec3(lights[i].color.x, lights[i].color.y, lights[i].color.z);
         vec3 ambient = 0.01 * lightColor; // low influence

		 vec4 fragPosModelSpace = vec4(inFragPos, 1.0);
         mat4 lightClip = lightMatricies[i].projectionMatrix * lightMatricies[i].viewMatrix;
		 vec4 fragPosLightSpace = lightClip * fragPosModelSpace;

		 // shadow factor computation
         float shadowFactor = shadowPCF(i, fragPosLightSpace, 4, normal, fragToLightDir);

         // spotlight cutoff
         if (theta > cos(outerConeRads)){
             float intensity;
         if (theta > cos(innerConeRads)){
             intensity = 1.0;
         } else {
             intensity = (theta - cos(outerConeRads)) / (cos(innerConeRads) - cos(outerConeRads));
         }
         intensity *= lights[i].intensity; // multiply it by the base intensity
         
         // attenuation calculation
         float lightDistance = length(inFragPos - lightPos);
         float attenuation = 1.0 / (constAttenuation + linAttenuation * lightDistance + quadAttenuation * (lightDistance * lightDistance));

         // diffuse lighting using lambertian reflectance
         float diff = max(dot(fragToLightDir, fragToLightDir), 0.0);
         diffuse += lightColor * diff * intensity * attenuation;

         // cook-torrance specular lighting
         float roughness = metallicRoughness.g; // roughness is stored in the green channel for gltf
         float metallic = metallicRoughness.b; // metallic is stored in the blue channel for gltf
         vec3 F0 = mix(vec3(0.04), lightColor, metallic);
         float cookTorranceSpecular = cookTorranceSpec(normal, fragToLightDir, inViewDir, roughness, F0); // fix
         //specular += lightColor * cookTorranceSpecular * attenuation;
         specular += lightColor * attenuation;
         }

    // final color calculation
    vec3 result = (ambient + (1.0 - shadowFactor) * (diffuse + specular)) * color;
    outColor = vec4(result, 1.0);
    }
}



