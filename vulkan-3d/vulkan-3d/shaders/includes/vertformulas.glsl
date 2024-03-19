vec4 getPos(mat4 proj, mat4 view, mat4 model, vec3 pos) {
	return proj * view * model * vec4(pos, 1.0);
}

vec3 getViewDir(mat4 view, mat4 model, vec3 pos) {
	vec3 worldCamPos = vec3(inverse(view)[3]);
	vec4 worldPos = model * vec4(pos, 1.0);
	return normalize(worldPos.xyz - worldCamPos);
}

mat3 getTBN(vec4 tangent, mat4 model, vec3 normal) {
	float handedness = tangent.w;

	mat3 normMat = transpose(inverse(mat3(model)));
	vec3 N = normalize(normMat * normal);
	vec3 T = normalize(normMat * tangent.xyz);
	T = normalize(T - dot(T, N) * N); // re orthogonalize tangent
	vec3 B = normalize(cross(N, T) * handedness);
	return mat3(T, B, N);

}
