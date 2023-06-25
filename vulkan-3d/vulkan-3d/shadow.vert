#version 460

layout (location = 0) in vec3 inPosition;

struct light {
    vec3 lPos;
    vec3 lColor;
    float lightIntensity;
    mat4 viewMatrix;
    mat4 modelMatrix;
    mat4 projectionMatrix;
};
layout (set=3, binding = 3) buffer LightBuffer {
	light lights[];
};

layout (location = 0) out vec4 fragPosLightSpace;

void main() {
if (lights.length() >= 1) {
 for (int i = 0; i < lights.length(); i++){ 
    fragPosLightSpace = lights[i].projectionMatrix * lights[i].viewMatrix * vec4(inPosition, 1.0);
    gl_Position = fragPosLightSpace;
}
}
}

