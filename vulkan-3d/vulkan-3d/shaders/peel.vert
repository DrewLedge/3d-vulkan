#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;

// individual rows of the instanced model matrix
layout(location = 3) in vec4 model1;
layout(location = 4) in vec4 model2;
layout(location = 5) in vec4 model3;
layout(location = 6) in vec4 model4;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outTexCoord;


layout(set = 1, binding = 4) uniform camBufferObject {
    mat4 view;
    mat4 proj;
} camUBO;

void main() {
	mat4 model = mat4(model1, model2, model3, model4);
    gl_Position = camUBO.proj * camUBO.view * model * vec4(inPosition, 1.0);
    outColor = inColor;
    outTexCoord = inTexCoord;
}
