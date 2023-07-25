#pragma once
#ifndef FORMULAS_H
#define FORMULAS_H
const float PI = 3.14159f;
class forms {
public:
	int rng(int m, int mm);
	struct vec3 {
		float x, y, z;
		vec3() : x(0.0f), y(0.0f), z(0.0f) {}
		vec3(float x, float y, float z) : x(x), y(y), z(z) {}

		vec3 operator+(const vec3& other) const {
			return vec3(x + other.x, y + other.y, z + other.z);

		}
		vec3& operator+=(float scalar) {
			x += scalar;
			y += scalar;
			z += scalar;
			return *this;
		}

		vec3 operator-(const vec3& other) const {
			return vec3(x - other.x, y - other.y, z - other.z);
		}
		vec3 operator*(float scalar) const {
			return vec3(x * scalar, y * scalar, z * scalar);
		}
		friend vec3 operator*(float scalar, const vec3& v) {
			return vec3(v.x * scalar, v.y * scalar, v.z * scalar);
		}
		vec3 operator*(const vec3& other) const {
			return vec3(x * other.x, y * other.y, z * other.z);
		}

		vec3 operator/(float scalar) const {
			return vec3(x / scalar, y / scalar, z / scalar);
		}
		friend vec3 operator/(float scalar, const vec3& v) {
			return vec3(scalar / v.x, scalar / v.y, scalar / v.z);
		}

		vec3& operator+=(const vec3& other) {
			x += other.x;
			y += other.y;
			z += other.z;
			return *this;
		}
		vec3& operator-=(const vec3& other) {
			x -= other.x;
			y -= other.y;
			z -= other.z;
			return *this;
		}
		bool operator==(const forms::vec3& other) const {
			const float epsilon = 0.00001f;
			return std::abs(x - other.x) < epsilon &&
				std::abs(y - other.y) < epsilon &&
				std::abs(z - other.z) < epsilon;
		}

		vec3 translate(float tx, float ty, float tz) const {
			return vec3(x + tx, y + ty, z + tz);
		}
		vec3 multiply(float sx, float sy, float sz) const {
			return vec3(x * sx, y * sy, z * sz);
		}

		static vec3 eulerToDir(const vec3& rotation) { // converts Euler rot to direction vector (right-handed coordinate system)
			// convert pitch and yaw from degrees to radians
			float pitch = rotation.x * (PI / 180.0f); // x rot
			float yaw = rotation.y * (PI / 180.0f); // y rot

			vec3 direction;
			direction.x = cos(yaw) * cos(pitch);
			direction.y = sin(pitch);
			direction.z = sin(yaw) * cos(pitch);
			return direction;
		}

		static vec3 getTargetVec(const vec3& pos, const vec3& rotation) { // computes target position from current position and rotation
			vec3 forwardDir = eulerToDir(rotation);
			return pos + forwardDir;
		}

		static vec3 dirToEuler(const vec3& direction) { // converts direction vector to Euler rot (right handed coordinate system)
			vec3 normalizedDir = direction.normalize();

			vec3 rotation;
			rotation.y = atan2(normalizedDir.z, normalizedDir.x) * (180.0 / PI); // yaw
			rotation.x = atan2(normalizedDir.y, sqrt(normalizedDir.x * normalizedDir.x + normalizedDir.z * normalizedDir.z)) * (180.0 / PI); // pitch
			rotation.z = 0; // roll
			return rotation;
		}

		static vec3 getForward(const vec3& camData) { // computes camera's forward direction (left handed coordinate system) (only use for camera)
			float pitch = camData.x;
			float yaw = camData.y;
			return vec3(
				-std::sin(yaw) * std::cos(pitch),
				-std::sin(pitch),
				std::cos(yaw) * std::cos(pitch)
			);
		}

		static vec3 getRight(const vec3& camData) { // computes camera's right direction (left handed coordinate system) (only use for camera)
			vec3 forward = getForward(camData);
			vec3 up(0.0f, -1.0f, 0.0f);
			return forward.crossProd(up);
		}
		static vec3 getUp(const vec3& camData) { // computes camera's up direction (left handed coordinate system) (only use for camera)
			vec3 forward = getForward(camData);
			vec3 right = getRight(camData);
			return right.crossProd(forward);
		}

		vec3 crossProd(vec3& v) const {
			return vec3(
				y * v.z - z * v.y,
				z * v.x - x * v.z,
				x * v.y - y * v.x
			);
		}
		float dotProd(const vec3& v) const {
			return x * v.x + y * v.y + z * v.z;
		}

		vec3 normalize() const {
			float length = std::sqrt(x * x + y * y + z * z);
			return vec3(x / length, y / length, z / length);
		}

		static vec3 toRads(const vec3& v) {
			return vec3(
				v.x * PI / 180.0f,
				v.y * PI / 180.0f,
				v.z * PI / 180.0f
			);
		}
		static float toDeg(const float radian) {
			return radian * (180.0f / PI);
		}
		static float toRad(const float degree) {
			return degree * (PI / 180.0f);
		}
	};
	struct vec2 {
		float x, y;
		bool operator==(const forms::vec2& other) const {
			const float epsilon = 0.00001f;
			return std::abs(x - other.x) < epsilon &&
				std::abs(y - other.y) < epsilon;
		}
		vec2(float x, float y) : x(x), y(y) {}
	};
	struct mat4 {
		float m[4][4];
		mat4() {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					m[i][j] = (i == j) ? 1.0f : 0.0f;
				}
			}
		}
		mat4(const mat4& other) {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					m[i][j] = other.m[i][j];
				}
			}
		}
		bool operator==(const mat4& other) const {
			const float epsilon = 0.00001f;
			bool equal = true;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					equal = equal && std::abs(m[i][j] - other.m[i][j]) < epsilon;
				}
			}
			return equal;
		}
		mat4 operator*(const mat4& other) const {
			mat4 result;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					result.m[i][j] = 0;
					for (int k = 0; k < 4; k++) {
						result.m[i][j] += m[i][k] * other.m[k][j];
					}
				}
			}
			return result;
		}
		mat4& operator *=(const mat4& other) {
			mat4 temp;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					temp.m[i][j] = 0;
					for (int k = 0; k < 4; k++) {
						temp.m[i][j] += m[i][k] * other.m[k][j];
					}
				}
			}
			*this = temp;
			return *this;
		}

		static mat4 translate(const vec3 t) {
			mat4 result;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					if (i == j) { //diagonal
						result.m[i][j] = 1.0f;
					}
					else {
						result.m[i][j] = 0.0f;
					}
				}
			}
			result.m[0][3] = t.x;
			result.m[1][3] = t.y;
			result.m[2][3] = t.z;
			return result;
		}
		static mat4 scale(const vec3 s) {
			mat4 result;
			result.m[0][0] = s.x;
			result.m[1][1] = s.y;
			result.m[2][2] = s.z;
			result.m[3][3] = 1.0f;
			return result;
		}
		static mat4 rotate(const vec3 s) {
			mat4 result;
			float radX = s.x * (PI / 180.0f);
			float radY = s.y * (PI / 180.0f);
			float radZ = s.z * (PI / 180.0f);

			mat4 rotX;
			rotX.m[1][1] = cosf(radX);
			rotX.m[1][2] = -sinf(radX);
			rotX.m[2][1] = sinf(radX);
			rotX.m[2][2] = cosf(radX);

			mat4 rotY;
			rotY.m[0][0] = cosf(radY);
			rotY.m[0][2] = sinf(radY);
			rotY.m[2][0] = -sinf(radY);
			rotY.m[2][2] = cosf(radY);

			mat4 rotZ;
			rotZ.m[0][0] = cosf(radZ);
			rotZ.m[0][1] = -sinf(radZ);
			rotZ.m[1][0] = sinf(radZ);
			rotZ.m[1][1] = cosf(radZ);

			result = rotZ * rotY * rotX;
			return result;
		}
		vec3 vecMatrix(const vec3& vec) const {
			float x = m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z + m[0][3]; //x is set to the dot product of the first row of the matrix and the vector
			float y = m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z + m[1][3];
			float z = m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z + m[2][3];
			float w = m[3][0] * vec.x + m[3][1] * vec.y + m[3][2] * vec.z + m[3][3];
			if (w != 1.0f && w != 0.0f) {
				x /= w;
				y /= w;
				z /= w;
			}
			return vec3(x, y, z);
		}
		static mat4 perspective(float fov, float aspect_ratio, float near_clip, float far_clip) {
			mat4 result;
			float radians = fov * (PI / 360.0f);
			float f = 1.0f / tanf(radians);
			result.m[0][0] = f / aspect_ratio;
			result.m[1][1] = f;
			result.m[2][2] = far_clip / (far_clip - near_clip);
			result.m[2][3] = -(far_clip * near_clip) / (far_clip - near_clip);
			result.m[3][2] = 1.0f;
			result.m[3][3] = 0.0f;
			return result;
		}

		static mat4 modelMatrix(vec3 trans, vec3 rot, vec3 s) {
			mat4 result = mat4::scale(s)
				* mat4::rotate(rot)
				* mat4::translate(trans);
			return result;
		}
		static mat4 inverseMatrix(mat4 m) {
			//formula from: https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html
			mat4 result;

			// det=a(ei−fh)−b(di−fg)+c(dh−eg)
			float cofactor0 = m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1];
			float cofactor1 = m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2];
			float cofactor2 = m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0];
			float det = m.m[0][0] * cofactor0 + m.m[0][1] * cofactor1 + m.m[0][2] * cofactor2;

			// check if the determinant is non-zero
			if (det != 0.0f) { // if not zero, then the matrix is invertible
				float invDet = 1.0f / det;

				// calculate the inverse of the upper-left 3x3 submatrix
				result.m[0][0] = invDet * cofactor0;
				result.m[1][0] = invDet * cofactor1;
				result.m[2][0] = invDet * cofactor2;
				result.m[0][1] = invDet * (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]);
				result.m[1][1] = invDet * (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]);
				result.m[2][1] = invDet * (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]);
				result.m[0][2] = invDet * (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]);
				result.m[1][2] = invDet * (m.m[0][2] * m.m[1][0] - m.m[0][0] * m.m[1][2]);
				result.m[2][2] = invDet * (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]);

				// calculate the inverse translation vector
				result.m[3][0] = -result.m[0][0] * m.m[3][0] - result.m[1][0] * m.m[3][1] - result.m[2][0] * m.m[3][2];
				result.m[3][1] = -result.m[0][1] * m.m[3][0] - result.m[1][1] * m.m[3][1] - result.m[2][1] * m.m[3][2];
				result.m[3][2] = -result.m[0][2] * m.m[3][0] - result.m[1][2] * m.m[3][1] - result.m[2][2] * m.m[3][2];
			}
			else {
				return m; //if not invertible, return original matrix
			}
			result.m[0][3] = 0.0f;
			result.m[1][3] = 0.0f;
			result.m[2][3] = 0.0f;
			result.m[3][3] = 1.0f;

			return result;
		}

		static mat4 viewMatrix(const vec3& position, const vec3& rotation) {
			mat4 result;
			result = mat4::rotate(rotation)
				* mat4::translate(position * -1);
			return result;
		}
		static mat4 lookAt(const vec3& eye, const vec3& target, const vec3& up) {
			vec3 f = (target - eye).normalize(); // forward vector
			vec3 r = up.crossProd(f).normalize(); // right vector
			vec3 u = f.crossProd(r); // up vector

			mat4 result;
			result.m[0][0] = r.x;
			result.m[0][1] = r.y;
			result.m[0][2] = r.z;
			result.m[0][3] = -r.dotProd(eye);
			result.m[1][0] = u.x;
			result.m[1][1] = u.y;
			result.m[1][2] = u.z;
			result.m[1][3] = -u.dotProd(eye);
			result.m[2][0] = f.x;
			result.m[2][1] = f.y;
			result.m[2][2] = f.z;
			result.m[2][3] = -f.dotProd(eye);
			result.m[3][0] = 0.0f;
			result.m[3][1] = 0.0f;
			result.m[3][2] = 0.0f;
			result.m[3][3] = 1.0f;
			return result;
		}
		static mat4 depthRangeMatrix() { // used for shadow mapping and getting the correct projection matrix
			mat4 result;
			vec3 scaleVector(1.0f, 1.0f, 0.5f);
			vec3 translateVector(0.0f, 0.0f, 0.5f);
			result = mat4::scale(scaleVector) * mat4::translate(translateVector);
			return result;
		}
	};
};

#endif
;
