#pragma once
#ifndef FORMULAS_H
#define FORMULAS_H
const float PI = 3.14159f;
class formulas {
public:
	int rng(int m, int mm);
	struct Vector3 {
		float x, y, z;
		Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
		Vector3 operator+(const Vector3& other) const {
			return Vector3(x + other.x, y + other.y, z + other.z);
		}

		Vector3 operator-(const Vector3& other) const {
			return Vector3(x - other.x, y - other.y, z - other.z);
		}

		Vector3 operator*(float scalar) const {
			return Vector3(x * scalar, y * scalar, z * scalar);
		}
		friend Vector3 operator*(float scalar, const Vector3& v) {
			return Vector3(v.x * scalar, v.y * scalar, v.z * scalar);
		}
		Vector3& operator+=(const Vector3& other) {
			x += other.x;
			y += other.y;
			z += other.z;
			return *this;
		}
		Vector3& operator-=(const Vector3& other) {
			x -= other.x;
			y -= other.y;
			z -= other.z;
			return *this;
		}
		bool operator==(const formulas::Vector3& other) const {
			const float epsilon = 0.00001f;
			return std::abs(x - other.x) < epsilon &&
				std::abs(y - other.y) < epsilon &&
				std::abs(z - other.z) < epsilon;
		}
		Vector3 translate(float tx, float ty, float tz) const {
			return Vector3(x + tx, y + ty, z + tz);
		}
		Vector3 multiply(float sx, float sy, float sz) const {
			return Vector3(x * sx, y * sy, z * sz);
		}
		Vector3 getForward(Vector3 camData) const {
			float pitch = camData.x;
			float yaw = camData.y;
			return Vector3(cos(yaw) * cos(pitch), sin(pitch), sin(yaw) * cos(pitch));
		}
		Vector3 getRight(Vector3 camData) const {
			Vector3 forward = getForward(camData);
			Vector3 up(0.0f, 1.0f, 0.0f);
			return forward.crossProd(up);
		}
		Vector3 crossProd(Vector3 v) const {
			return Vector3(
				y * v.z - z * v.y,
				z * v.x - x * v.z,
				x * v.y - y * v.x
			);
		}
	};
	struct Vector2 {
		float x, y;
		bool operator==(const formulas::Vector2& other) const {
			const float epsilon = 0.00001f;
			return std::abs(x - other.x) < epsilon &&
				std::abs(y - other.y) < epsilon;
		}
		Vector2(float x, float y) : x(x), y(y) {}
	};
	struct Matrix4 {
		float m[4][4];
		Matrix4() {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					m[i][j] = (i == j) ? 1.0f : 0.0f;
				}
			}
		}
		Matrix4(const Matrix4& other) {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					m[i][j] = other.m[i][j];
				}
			}
		}

		Matrix4 add(const Matrix4& other) const { //add two matrices together
			Matrix4 result;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					result.m[i][j] = m[i][j] + other.m[i][j];
				}
			}
			return result;
		}
		Matrix4 subtract(const Matrix4& other) const {
			Matrix4 result;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					result.m[i][j] = m[i][j] - other.m[i][j];
				}
			}
			return result;
		}
		Matrix4 multiply(const Matrix4& other) const { //multiply two matrices together
			Matrix4 result;
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
		static Matrix4 translate(float tx, float ty, float tz) {
			Matrix4 result;
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
			result.m[0][3] = tx;
			result.m[1][3] = ty;
			result.m[2][3] = tz;
			return result;
		}
		static Matrix4 scale(float sx, float sy, float sz) {
			Matrix4 result;
			result.m[0][0] = sx;
			result.m[1][1] = sy;
			result.m[2][2] = sz;
			result.m[3][3] = 1.0f;
			return result;
		}
		static Matrix4 rotate(float angleX, float angleY, float angleZ) {
			Matrix4 result;
			float radX = angleX * (PI / 180.0f);
			float radY = angleY * (PI / 180.0f);
			float radZ = angleZ * (PI / 180.0f);

			Matrix4 rotX;
			rotX.m[1][1] = cosf(radX);
			rotX.m[1][2] = -sinf(radX);
			rotX.m[2][1] = sinf(radX);
			rotX.m[2][2] = cosf(radX);

			Matrix4 rotY;
			rotY.m[0][0] = cosf(radY);
			rotY.m[0][2] = sinf(radY);
			rotY.m[2][0] = -sinf(radY);
			rotY.m[2][2] = cosf(radY);

			Matrix4 rotZ;
			rotZ.m[0][0] = cosf(radZ);
			rotZ.m[0][1] = -sinf(radZ);
			rotZ.m[1][0] = sinf(radZ);
			rotZ.m[1][1] = cosf(radZ);

			result = rotZ.multiply(rotY).multiply(rotX);
			return result;
		}
		Vector3 vecmatrix(const Vector3& vec) const {
			float x = m[0][0] * vec.x + m[0][1] * vec.y + m[0][2] * vec.z + m[0][3]; //x is set to the dot product of the first row of the matrix and the vector
			float y = m[1][0] * vec.x + m[1][1] * vec.y + m[1][2] * vec.z + m[1][3];
			float z = m[2][0] * vec.x + m[2][1] * vec.y + m[2][2] * vec.z + m[2][3];
			float w = m[3][0] * vec.x + m[3][1] * vec.y + m[3][2] * vec.z + m[3][3];
			if (w != 1.0f && w != 0.0f) {
				x /= w;
				y /= w;
				z /= w;
			}
			return Vector3(x, y, z);
		}
		static Matrix4 perspective(float fov, float aspect_ratio, float near_clip, float far_clip) {
			Matrix4 result;
			float f = 1.0f / tanf(fov * (PI / 360.0f));
			result.m[0][0] = f / aspect_ratio;
			result.m[1][1] = f;
			result.m[2][2] = far_clip / (far_clip - near_clip);
			result.m[2][3] = -(far_clip * near_clip) / (far_clip - near_clip);
			result.m[3][2] = 1.0f;
			result.m[3][3] = 0.0f;
			return result;
		}

		static Matrix4 worldmatrix(float tx, float ty, float tz, float rx, float ry, float rz, float sx, float sy, float sz) {
			// t = translation, r = rotation, s = scale
			Matrix4 result = Matrix4::translate(tx, ty, tz)
				.multiply(Matrix4::rotate(rx, ry, rz))
				.multiply(Matrix4::scale(sx, sy, sz));
			return result;
		}
		static Matrix4 inverseMatrix(Matrix4 m) {
			//formula from: https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html
			Matrix4 result;

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


		static Matrix4 viewmatrix(const Vector3& position, const Vector3& rotation) {
			Matrix4 result;
			result = Matrix4::rotate(rotation.x, rotation.y, rotation.z)
				.multiply(Matrix4::translate(-position.x, -position.y, -position.z));
			return result;
		}

	};
};

#endif
;
