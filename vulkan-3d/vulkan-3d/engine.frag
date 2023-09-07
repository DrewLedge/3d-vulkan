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

vec3 lightDirection;

float shadowPCF(int lightIndex, vec4 fragPosLightSpace, int kernelSize) { // get the PCF shadow factor (used for softer shadows)
    int halfSize = kernelSize / 2;
    fragPosLightSpace.xyz /= fragPosLightSpace.w;
    float shadow = 0.0;

    // transform to [0,1] range
    vec3 projCoords = fragPosLightSpace.xyz * 0.5 + 0.5;

    // calculate texel size based on shadow map dimensions
    vec2 texelSize = 1.0 / textureSize(shadowMapSamplers[lightIndex], 0);

    // loop through the PCF kernel
    for(int x = -halfSize; x <= halfSize; ++x) {
        for(int y = -halfSize; y <= halfSize; ++y) {
            // calculate the texel coordinates for sampling
            vec2 sampleCoords = projCoords.xy + vec2(x, y) * texelSize;

            // sample the depth from shadow map
            float pcfDepth = texture(shadowMapSamplers[lightIndex], sampleCoords).r;

            // perform depth comparison manually
            shadow += (fragPosLightSpace.z > pcfDepth) ? 1.0 : 0.0;
        }
    }

    // normalize the shadow factor
    shadow /= float(kernelSize * kernelSize);

    return shadow;
}


void main() {
float shinyness=32.0f;
float PI = acos(-1.0);
if (lights.length() >= 1) {
    vec4 sampled = texture(texSamplers[inTexIndex], inTexCoord); // diffuse map
    vec4 sampledSpec = texture(texSamplers[inTexIndex + 1], inTexCoord); // specular map
    vec4 sampledNorm = texture(texSamplers[inTexIndex + 2], inTexCoord); // normal map

    vec3 normal = normalize(sampledNorm.xyz * 2.0 - 1.0);
    vec3 ambient = 0.1 * sampled.rgb; // low influence

    vec3 diffuse = vec3(0.0); 
    vec3 specular = vec3(0.0);

    for (int i = 0; i < lights.length(); i++){ 
         float innerConeRads = lights[i].innerConeAngle * (PI/180.0f);
         float outerConeRads = lights[i].outerConeAngle * (PI/180.0f);

		 // convert light struct to vec3s so I can use them in calculations
		 vec3 lightPos = vec3(lights[i].pos.x, lights[i].pos.y, lights[i].pos.z);
         vec3 targetVec = vec3(lights[i].targetVec.x, lights[i].targetVec.y, lights[i].targetVec.z);
         vec3 lightDirection = normalize(targetVec - lightPos);
         vec3 fragToLightDirection = normalize(inFragPos - lightPos);
         float theta = dot(lightDirection, fragToLightDirection);
         vec3 lightColor = vec3(lights[i].color.x, lights[i].color.y, lights[i].color.z);

		 // shadow factor computation:
		 vec4 fragPosModelSpace = vec4(inFragPos, 1.0);
         mat4 lightClip = lightMatricies[i].projectionMatrix * lightMatricies[i].viewMatrix;
		 vec4 fragPosLightSpace = lightClip * fragPosModelSpace;

		 // shadow factor computation:
		 float shadowFactor =shadowPCF(i, fragPosLightSpace, 3);

		 // spotlight cutoff
         if(theta > cos(outerConeRads)){
             float intensity;
         if(theta > cos(innerConeRads)){
             intensity = 1.0;
         } else {
             intensity = (theta - cos(outerConeRads)) / (cos(innerConeRads) - cos(outerConeRads));
         }

         // diffuse lighting
         vec3 lightDir = normalize(lightPos - inFragPos);
         float diff = max(dot(normal, lightDir), 0.0);
         diffuse += lightColor * diff * intensity;

         // specular lighting
         vec3 viewDir = normalize(inCamPos - inFragPos);
         vec3 reflectDir = reflect(-lightDir, normal); 
         float spec = pow(max(dot(viewDir, reflectDir), 0.0), sampledSpec.a); // sampledSpec.a is used as the shininess factor
         specular += lightColor * spec * intensity * shadowFactor; // multiplying with shadow factor for shadow contribution
        }
    }

    vec3 result = ambient + diffuse * sampled.rgb + specular * sampledSpec.rgb;

    outColor = vec4(result, sampled.a);
}

}


