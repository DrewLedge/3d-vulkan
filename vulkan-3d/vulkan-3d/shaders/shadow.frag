#version 460

layout(location = 0) in vec3 inLightPos;
layout(location = 1) in vec3 inLightTarget;
layout(location = 2) in vec3 inFragPos;
layout(location = 3) in float inConeRadsCos;

void main() {
	vec3 spotDirection = normalize(inLightPos - inLightTarget);
	vec3 fragToLightDir = normalize(inLightPos - inFragPos);
	float theta = dot(spotDirection, fragToLightDir);

	// spotlight cutoff
	if (theta > inConeRadsCos) {
		gl_FragDepth = gl_FragCoord.z;
    } else {
		discard;
    }
}
