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
struct light {
    formsVec3 lPos;
    formsVec3 lColor;
    formsVec3 rot;
    float FOV;
    float lightIntensity;
    mat4 viewMatrix;
    mat4 modelMatrix;
    mat4 projectionMatrix;
};

layout (set=3, binding = 3) buffer LightBuffer {
	light lights[];
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

void main() {
if (lights.length() >= 1) {
    vec4 sampled = texture(texSamplers[inTexIndex], inTexCoord); // diffuse map
    vec4 sampledSpec = texture(texSamplers[inTexIndex + 1], inTexCoord); // specular map
    vec4 sampledNorm = texture(texSamplers[inTexIndex + 2], inTexCoord); // normal map

    vec3 normal = normalize(inNormal + sampledNorm.xyz * 2.0 - 1.0); 
    vec3 ambient = 0.1 * sampled.rgb; // low influence

    vec3 diffuse = vec3(0.0); 
    vec3 specular = vec3(0.0);

    for (int i = 0; i < lights.length(); i++){ 
        // convert light struct to vec3s so I can use them in calculations
        vec3 lightPos = vec3(lights[i].lPos.x, lights[i].lPos.y, lights[i].lPos.z);
        vec3 lightColor = vec3(lights[i].lColor.x, lights[i].lColor.y, lights[i].lColor.z);

        // directional lighting:
       vec3 lightDirection = normalize(-lightPos);


        // shadow factor computation:
        vec4 fragPosLightSpace = lights[i].viewMatrix * lights[i].modelMatrix * vec4(inFragPos, 1.0); // the position of the fragment in light space
        float shadowFactor = textureProj(shadowMapSamplers[i], fragPosLightSpace); 

        // blinn-Phong lighting model:
        float diff = max(dot(normal, lightDirection), 0.0); // represents the amount of light hitting the surface from that direction
        diffuse += lightColor * diff * lights[i].lightIntensity * shadowFactor; // modulate with shadow factor

        vec3 halfwayDir = normalize(lightDirection + inViewDir); 
        float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0); 
        specular += lightColor * sampledSpec.rgb * spec * lights[i].lightIntensity * shadowFactor; // adds the specular component to the final color and modulates with shadow factor
    }

    vec3 result = ambient + diffuse + specular; 
    outColor = vec4(result, 1.0) * inAlpha;
}
}

