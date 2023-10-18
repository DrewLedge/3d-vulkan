#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shadow_samplers : enable

layout(set = 1, binding = 1) uniform sampler2D texSamplers[];

layout(set = 4, binding = 4) uniform sampler2DShadow shadowMapSamplers[];

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

layout(location = 0) in vec4 fragColor; // base color data from vertex attributes
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) flat in uint inTexIndex;
layout(location = 3) flat in uint inModelIndex;
layout(location = 4) in vec3 inFragPos;
layout(location = 5) in vec3 inViewDir;

layout(location = 0) out vec4 outColor; 
float PI = acos(-1.0);



// get the PCF shadow factor (used for softer shadows)
float shadowPCF(int lightIndex, vec4 fragPosLightSpace, int kernelSize, vec3 norm, vec3 lightDir) {  
    int halfSize = kernelSize / 2;
    float shadow = 0.0;
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    // calculate texel size based on shadow map dimensions
    vec2 texelSize = 1.0 / textureSize(shadowMapSamplers[lightIndex], 0);

    // loop through the PCF kernel
    for(int x = -halfSize; x <= halfSize; ++x) {
        for(int y = -halfSize; y <= halfSize; ++y) {
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

vec3 cookTorrance(vec3 N, vec3 L, vec3 V, vec3 albedo, float metallic, float roughness) {
    float alpha = roughness * roughness;
    
    // compute halfway vector
    vec3 H = normalize(V + L);
    
    // compute the geometric term (GGX)
    float NdotH = max(dot(N, H), 0.0);
    float NdotV = max(dot(N, V), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float G = min(1.0, min((2.0 * NdotH * NdotV) / VdotH, (2.0 * NdotH * dot(N, L)) / VdotH));
    
    // compute the roughness term (GGX)
    float D = alpha * alpha / (PI * pow(NdotH * NdotH * (alpha * alpha - 1.0) + 1.0, 2.0));
    
    // compute the Fresnel term (schlick approximation)
    vec3 F = albedo + (1.0 - albedo) * pow(1.0 - VdotH, 5.0);
    
    vec3 specular = (D * G * F) / (4.0 * NdotV * dot(N, L));
    vec3 diffuse = (1.0 - metallic) * albedo * (1.0 / PI);
    
    return (diffuse + specular);
}



void main() {
    vec4 albedo = texture(texSamplers[inTexIndex], inTexCoord);
    vec4 metallicRoughness = texture(texSamplers[inTexIndex + 1], inTexCoord);
    vec4 normalMap = texture(texSamplers[inTexIndex + 2], inTexCoord);

    vec3 normal = normalize(normalMap.xyz * 2.0 - 1.0);
    vec3 color = albedo.rgb;

    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);

    float shadowFactor = 0.0;
    vec3 ambient = vec3(0.0);

    vec3 brdf;

    for (int i = 0; i < lights.length(); i++) { // spotlight
        mat4 lightView = lightMatricies[i].viewMatrix;
        mat4 lightProj = lightMatricies[i].projectionMatrix;

        float innerConeRads = lights[i].innerConeAngle * (PI / 180.0f);
        float outerConeRads = lights[i].outerConeAngle * (PI / 180.0f);
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
        ambient = 0.1 * lightColor; // low influence

        
        // spotlight cutoff
        if (theta > cos(outerConeRads)) { // if inside the cone, calculate lighting
            vec4 fragPosLightSpace = lightProj * lightView * vec4(inFragPos, 1.0);
            shadowFactor = shadowPCF(i, fragPosLightSpace, 4, normal, fragToLightDir);

            float intensity;
            if (theta > cos(innerConeRads)) {
                intensity = 1.0; 
            } else {
                intensity = (theta - cos(outerConeRads)) / (cos(innerConeRads) - cos(outerConeRads));
            }

            // attenuation calculation
            float lightDistance = length(inFragPos - lightPos);
            float attenuation = 1.0 / (constAttenuation + linAttenuation * lightDistance + quadAttenuation * (lightDistance * lightDistance));

            // diffuse lighting using lambertian reflectance
            float diff = max(dot(fragToLightDir, normal), 0.0);
            diffuse += lightColor * diff * intensity * attenuation;

            // cook-torrance specular lighting WIP
            float roughness = metallicRoughness.g; // roughness is stored in the green channel for gltf
            float metallic = metallicRoughness.b;  // metallic is stored in the blue channel for gltf
            vec3 F0 = mix(vec3(0.04), lightColor, metallic);
            // specular += lightColor * cookTorranceSpecular * attenuation;
            specular += lightColor * attenuation;

            brdf = cookTorrance(normal, fragToLightDir, inViewDir, color, metallic, roughness);
        }
    }
    
    // final color calculation
    vec3 result = (ambient + shadowFactor * brdf);
    outColor = vec4(result, 1.0);

    //outColor = vec4(shadowFactor, shadowFactor * 0.5, shadowFactor * 0.5, 1.0);
    
}



