#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(set = 1, binding = 1) uniform sampler2D texSamplers[];

struct light { // omnidirectional light
	float lPos[3];
	float lColor[3];
	float lightIntensity;
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
vec3 lightPos;
vec3 lightColor;
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
        //convert my light struct to vec3s so I can use them in calculations
        lightPos = vec3(lights[i].lPos[0], lights[i].lPos[1], lights[i].lPos[2]);
        lightColor = vec3(lights[i].lColor[0], lights[i].lColor[1], lights[i].lColor[2]);

        // blinn-phong lighting model:
        vec3 lightDir = normalize(lightPos - inFragPos); // calculate the light direction based on the light position and the fragment position
        float diff = max(dot(normal, lightDir), 0.0); // represents the amount of light hitting the surface from that direction
        diffuse += lightColor * diff * lights[i].lightIntensity;

        // calculate the halfway vector
        vec3 halfwayDir = normalize(lightDir + inViewDir); // represents the direction halfway between the light and the view direction
        float spec = pow(max(dot(normal, halfwayDir), 0.0), 32.0); // calculate the specular intensity
        specular += lightColor * sampledSpec.rgb * spec * lights[i].lightIntensity; // adds the specular component to the final color
    }

    vec3 result = ambient + diffuse + specular; // adds up the ambient, diffuse, and specular contributions to obtain the final color
    outColor = vec4(result, 1.0) * inAlpha;
}
}
