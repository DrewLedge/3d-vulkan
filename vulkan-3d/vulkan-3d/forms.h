#pragma once
#include <iostream>
#ifndef FORMULAS_H
#define FORMULAS_H
const float PI = acos(-1.0f);
class forms {
public:
	int rng(int m, int mm);
	struct gen {
		static float getPercent(const float value, const float max) {
			return (value / max) * 100.0f;
		}
	};
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

		friend std::ostream& operator<<(std::ostream& os, const vec3& v) {
			os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
			return os;
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
	struct vec4 {
		float x, y, z, w;

		vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
		vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

		bool operator==(const vec4& other) const {
			const float epsilon = 0.00001f;
			return std::abs(x - other.x) < epsilon &&
				std::abs(y - other.y) < epsilon &&
				std::abs(z - other.z) < epsilon &&
				std::abs(w - other.w) < epsilon;
		}

		bool operator!=(const vec4& other) const {
			return !(*this == other);
		}
		vec4 operator+(const vec4& other) const {
			return vec4(x + other.x, y + other.y, z + other.z, w + other.w);
		}
		vec4 operator-(const vec4& other) const {
			return vec4(x - other.x, y - other.y, z - other.z, w - other.w);
		}
		vec4 operator*(float scalar) const {
			return vec4(x * scalar, y * scalar, z * scalar, w * scalar);
		}
		vec4 operator/(float scalar) const {
			return vec4(x / scalar, y / scalar, z / scalar, w / scalar);
		}
		vec4& operator+=(const vec4& other) {
			x += other.x;
			y += other.y;
			z += other.z;
			w += other.w;
			return *this;
		}
		vec4& operator-=(const vec4& other) {
			x -= other.x;
			y -= other.y;
			z -= other.z;
			w -= other.w;
			return *this;
		}
		vec4& operator*=(float scalar) {
			x *= scalar;
			y *= scalar;
			z *= scalar;
			w *= scalar;
			return *this;
		}
		vec4& operator/=(float scalar) {
			x /= scalar;
			y /= scalar;
			z /= scalar;
			w /= scalar;
			return *this;
		}
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
		mat4(float zero) {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					m[i][j] = 0.0f;
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


		friend vec3 operator *(const mat4& mat, const vec3& vec) {
			float x = mat.m[0][0] * vec.x + mat.m[0][1] * vec.y + mat.m[0][2] * vec.z + mat.m[0][3];
			float y = mat.m[1][0] * vec.x + mat.m[1][1] * vec.y + mat.m[1][2] * vec.z + mat.m[1][3];
			float z = mat.m[2][0] * vec.x + mat.m[2][1] * vec.y + mat.m[2][2] * vec.z + mat.m[2][3];
			float w = mat.m[3][0] * vec.x + mat.m[3][1] * vec.y + mat.m[3][2] * vec.z + mat.m[3][3];
			if (w != 1.0f && w != 0.0f) {
				x /= w;
				y /= w;
				z /= w;
			}
			return vec3(x, y, z);
		}

		static mat4 translate(const vec3 t) {
			mat4 result;
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
			float radX = s.x * (PI / 180.0f); // convert to radians
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
		static mat4 rotateQ(const vec4 q) { // quaternian rotation (right handed - column major)
			// formula from: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
			mat4 result;

			float w = q.w;
			float x = q.x;
			float y = q.y;
			float z = q.z;

			result.m[0][0] = 1.0f - 2.0f * y * y - 2.0f * z * z;
			result.m[0][1] = 2.0f * x * y - 2.0f * z * w;
			result.m[0][2] = 2.0f * x * z + 2.0f * y * w;
			result.m[0][3] = 0.0f;

			result.m[1][0] = 2.0f * x * y + 2.0f * z * w;
			result.m[1][1] = 1.0f - 2.0f * x * x - 2.0f * z * z;
			result.m[1][2] = 2.0f * y * z - 2.0f * x * w;
			result.m[1][3] = 0.0f;

			result.m[2][0] = 2.0f * x * z - 2.0f * y * w;
			result.m[2][1] = 2.0f * y * z + 2.0f * x * w;
			result.m[2][2] = 1.0f - 2.0f * x * x - 2.0f * y * y;
			result.m[2][3] = 0.0f;

			result.m[3][3] = 1.0f;

			return result;
		}

		static mat4 projection(float fov, float aspect, float nearPlane, float farPlane) {
			mat4 result(0);
			float fovRad = fov * (PI / 180.0f);
			float tanHalf = tan(fovRad * 0.5f);

			// column major
			result.m[0][0] = 1.0f / (aspect * tanHalf);
			result.m[1][1] = -1.0f / tanHalf;
			result.m[2][2] = farPlane / (farPlane - nearPlane);
			result.m[2][3] = -(2 * farPlane * nearPlane) / (farPlane - nearPlane);
			result.m[3][2] = 1.0f;
			result.m[3][3] = 0.0f;

			return result;
		}
		static mat4 spotPerspective(float verticalFov, float aspectRatio, float n, float f) {
			float fovRad = verticalFov * 2.0f * PI / 360.0f; // convert to radians
			float focalLength = 1.0f / tan(fovRad / 2.0f);
			float x = focalLength / aspectRatio;
			float y = -focalLength;
			float a = f / (n - f);
			float b = (f * n) / (n - f);


			mat4 proj;
			proj.m[0][0] = x;
			proj.m[1][1] = y;
			proj.m[2][2] = a;
			proj.m[2][3] = b;
			proj.m[3][2] = -1.0f;
			return proj;

		}

		static mat4 modelMatrix(vec3 trans, vec3 rot, vec3 s) {
			return mat4::scale(s) * mat4::rotate(rot) * mat4::translate(trans);;
		}


		static float submatrixDet(mat4 m, int excludeRow, int excludeCol) {
			float det, sub[3][3];
			int x = 0, y = 0;
			for (int i = 0; i < 4; i++) {
				if (i == excludeRow) continue;
				y = 0;
				for (int j = 0; j < 4; j++) {
					if (j == excludeCol) continue;
					sub[x][y] = m.m[j][i];
					y++;
				}
				x++;
			}
			// get the determinant of the submatrix
			det = sub[0][0] * (sub[1][1] * sub[2][2] - sub[2][1] * sub[1][2]) - sub[0][1] * (sub[1][0] * sub[2][2] - sub[2][0] * sub[1][2]) + sub[0][2] * (sub[1][0] * sub[2][1] - sub[2][0] * sub[1][1]);
			return det;
		}

		static mat4 inverseMatrix(mat4 m) {
			//formula from: https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html
			mat4 cofactor;
			float det = 0.0f;

			// calculate determinant
			for (int i = 0; i < 4; i++) {
				det += ((i % 2 == 0 ? 1 : -1) * m.m[i][0] * submatrixDet(m, 0, i));
			}

			if (det == 0.0f) { // if matrix is not invertible, return original matrix
				std::cerr << "Matrix not invertible!" << std::endl;
				return m;
			}

			float invDet = 1.0f / det;

			// calculate cofactor matrix
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					int sign = ((i + j) % 2 == 0) ? 1 : -1;
					cofactor.m[j][i] = sign * submatrixDet(m, i, j);
				}
			}

			// multiply by 1/det to get inverse
			mat4 inverse;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					inverse.m[i][j] = invDet * cofactor.m[i][j];
				}
			}

			return inverse;
		}

		static mat4 viewMatrix(const vec3& position, const vec3& rotation) {
			mat4 result;
			result = mat4::rotate(rotation)
				* mat4::translate(position);
			return result;
		}

		static vec3 getCamWorldPos(const mat4& viewMat) {
			mat4 invView = inverseMatrix(viewMat);
			vec3 cameraWorldPosition(invView.m[3][0], invView.m[3][1], invView.m[3][2]);
			return cameraWorldPosition;
		}

		static mat4 lookAt(const vec3& eye, const vec3& target, vec3& inputUpVector) {
			vec3 f = (target - eye).normalize(); // forward vector
			vec3 r = f.crossProd(inputUpVector).normalize(); // right vector
			vec3 u = r.crossProd(f).normalize(); // up vector
			mat4 result;

			result.m[0][0] = r.x;
			result.m[1][0] = r.y;
			result.m[2][0] = r.z;
			result.m[3][0] = -r.dotProd(eye);

			result.m[0][1] = u.x;
			result.m[1][1] = u.y;
			result.m[2][1] = u.z;
			result.m[3][1] = -u.dotProd(eye);

			// negate f due to right hand cord system
			result.m[0][2] = -f.x;
			result.m[1][2] = -f.y;
			result.m[2][2] = -f.z;
			result.m[3][2] = f.dotProd(eye);

			result.m[3][3] = 1.0f;
			return result.transpose();
		}

		mat4 transpose() const {
			mat4 result;
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					result.m[j][i] = m[i][j];
				}
			}
			return result;
		}
		static mat4 gltfToMat4(const std::vector<double>& vec) {
			mat4 result;
			int index = 0;

			for (int col = 0; col < 4; ++col) {
				for (int row = 0; row < 4; ++row) {
					result.m[row][col] = static_cast<float>(vec[index]);
					++index;
				}
			}
			return result;
		}
	};
};

#endif
;
