// Drew's Math Library (DML)

#pragma once
#include <iostream>
#ifndef DLM_H
#define DLM_H
const float PI = acos(-1.0f);
class dml {
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
		friend vec3 operator-(const vec3& v, float scalar) {
			return vec3(v.x - scalar, v.y - scalar, v.z - scalar);
		}
		friend vec3 operator+(const vec3& v, float scalar) {
			return vec3(v.x + scalar, v.y + scalar, v.z + scalar);
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
		vec3& operator*=(float scalar) {
			x *= scalar;
			y *= scalar;
			z *= scalar;
			return *this;
		}
		vec3& operator/=(float scalar) {
			x /= scalar;
			y /= scalar;
			z /= scalar;
			return *this;
		}
		vec3& operator*=(const vec3& other) {
			x *= other.x;
			y *= other.y;
			z *= other.z;
			return *this;
		}
		vec3& operator/=(const vec3& other) {
			x /= other.x;
			y /= other.y;
			z /= other.z;
			return *this;
		}

		bool operator==(const dml::vec3& other) const {
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
		float length() const {
			return std::sqrt(x * x + y * y + z * z);
		}

		vec3 normalize() const {
			float length = std::sqrt(x * x + y * y + z * z);

			// check for zero length to avoid division by zero
			if (length < std::numeric_limits<float>::epsilon()) {
				return vec3(0.0f, 0.0f, 0.0f);
			}

			return vec3(x / length, y / length, z / length);
		}


	};
	struct vec2 {
		float x, y;
		friend std::ostream& operator<<(std::ostream& os, const vec2& v) {
			os << "(" << v.x << ", " << v.y << ")";
			return os;
		}
		bool operator==(const dml::vec2& other) const {
			const float epsilon = 0.00001f;
			return std::abs(x - other.x) < epsilon &&
				std::abs(y - other.y) < epsilon;
		}
		vec2(float x, float y) : x(x), y(y) {}
		vec2 operator+(const vec2& other) const {
			return vec2(x + other.x, y + other.y);
		}

		vec2 operator-(const vec2& other) const {
			return vec2(x - other.x, y - other.y);
		}

		vec2 operator*(float scalar) const {
			return vec2(x * scalar, y * scalar);
		}

		vec2 operator*(const vec2& other) const {
			return vec2(x * other.x, y * other.y);
		}
		vec2 crossProd(const vec2& other) const {
			return vec2(x * other.y, y * other.x);
		}
	};
	struct vec4 {
		float x, y, z, w;

		vec4() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
		vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

		vec2 xy() const { return vec2(x, y); }
		vec3 xyz() const { return vec3(x, y, z); }

		vec4 normalize() const{
			float length = sqrt(x * x + y * y + z * z + w * w);
			if (length == 0) {
				return vec4(0.0f, 0.0f, 0.0f, 1.0f);
			}

			return vec4(x / length, y / length, z / length , w / length);
		}

		vec4 conjugate() const {
			return vec4(-x, -y, -z, w);
		}

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
		vec4 operator*(const vec4& other) const {
			float x1 = x, y1 = y, z1 = z, w1 = w;
			float x2 = other.x, y2 = other.y, z2 = other.z, w2 = other.w;

			return vec4(
				w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, // x
				w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, // y
				w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, // z
				w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2 // w
			);
		}

		float& operator[](size_t index) {
			// bounds check for the index
			if (index > 3) {
				throw std::out_of_range("Index out of range for vec4");
			}

			// select the component based on the index
			switch (index) {
			case 0: return x;
			case 1: return y;
			case 2: return z;
			case 3: return w;
			default: throw std::invalid_argument("Invalid index for vec4");
			}
		}

		const float& operator[](size_t index) const {
			// bounds check for the index
			if (index > 3) {
				throw std::out_of_range("Index out of range for vec4");
			}

			// select the component based on the index
			switch (index) {
			case 0: return x;
			case 1: return y;
			case 2: return z;
			case 3: return w;
			default: throw std::invalid_argument("Invalid index for vec4");
			}
		}


		friend std::ostream& operator<<(std::ostream& os, const vec4& v) {
			os << "( x:" << v.x << ", y:" << v.y << ", z:" << v.z << ", w:" << v.w << ")";
			return os;
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

		mat4 transpose() const {
			mat4 result;
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					result.m[j][i] = m[i][j];
				}
			}
			return result;
		}
	};
	// ------------------ VECTOR3 FORMULAS ------------------ //
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

	// ------------------ VECTOR4 FORMULAS ------------------ //
	static vec4 targetToQ(const vec3& position, const vec3& target) {
		vec3 up = { 0.0f, 1.0f, 0.0f };
		mat4 l = lookAt(target, position, up);
		return quatCast(l);
	}

	static vec4 inverseQ(const vec4& q) { // quaternion inversion
		float length = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;	
		if (length == 0) {
			return vec4(0.0f, 0.0f, 0.0f, 1.0f); // return identity quaternion
		}

		return vec4(-q.x / length, -q.y / length, -q.z / length, q.w / length);
	}

	static vec4 angleAxis(float angle, const vec3& axis) {
		vec3 normAxis = axis.normalize();

		// compute the sin and cos of half the angle
		float half = angle * 0.5f;
		float sinHalf = sin(half);
		float cosHalf = cos(half);

		// create the quaternion
		return vec4(normAxis.x * sinHalf, normAxis.y * sinHalf, normAxis.z * sinHalf, cosHalf);
	}

	// ------------------ MATRIX4 FORMULAS ------------------ // 
	static vec4 quatCast(const mat4& mat) {
		float trace = mat.m[0][0] + mat.m[1][1] + mat.m[2][2];
		vec4 quaternion;

		if (trace > 0.0f) { // if trace is positive
			float s = sqrt(trace + 1.0f); // compute scale factor
			quaternion.w = s * 0.5f; // quaternion scale part
			s = 0.5f / s;

			// get the quaternion vector parts
			quaternion.x = (mat.m[2][1] - mat.m[1][2]) * s;
			quaternion.y = (mat.m[0][2] - mat.m[2][0]) * s;
			quaternion.z = (mat.m[1][0] - mat.m[0][1]) * s;
		}
		else { // if trace is negative
			int i = 0;

			// find the greatest diagonal element to ensure better stability when computing square root
			if (mat.m[1][1] > mat.m[0][0]) i = 1;
			if (mat.m[2][2] > mat.m[i][i]) i = 2;
			int j = (i + 1) % 3;
			int k = (i + 2) % 3;

			float s = sqrt(mat.m[i][i] - mat.m[j][j] - mat.m[k][k] + 1.0f);
			quaternion[i] = s * 0.5f; // quaternion scale part
			s = 0.5f / s;

			// get the quaternion vector parts
			quaternion.w = (mat.m[k][j] - mat.m[j][k]) * s;
			quaternion[j] = (mat.m[j][i] + mat.m[i][j]) * s;
			quaternion[k] = (mat.m[k][i] + mat.m[i][k]) * s;
		}
		// normalize the quaternion
		float length = sqrt(quaternion.w * quaternion.w + quaternion.x * quaternion.x + quaternion.y * quaternion.y + quaternion.z * quaternion.z);
		quaternion /= length;

		// return the calculated quaternion
		return quaternion;
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

		result.m[0][0] = 1.0f - 2.0f * (y * y + z * z);
		result.m[0][1] = 2.0f * (x * y + z * w);
		result.m[0][2] = 2.0f * (x * z - y * w);
		result.m[0][3] = 0.0f;

		result.m[1][0] = 2.0f * (x * y - z * w);
		result.m[1][1] = 1.0f - 2.0f * (x * x + z * z);
		result.m[1][2] = 2.0f * (y * z + x * w);
		result.m[1][3] = 0.0f;

		result.m[2][0] = 2.0f * (x * z + y * w);
		result.m[2][1] = 2.0f * (y * z - x * w);
		result.m[2][2] = 1.0f - 2.0f * (x * x + y * y);
		result.m[2][3] = 0.0f;

		result.m[3][0] = 0.0f;
		result.m[3][1] = 0.0f;
		result.m[3][2] = 0.0f;
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
		return scale(s) * rotate(rot) * translate(trans);;
	}


	static float submatrixDet(mat4 m, int excludeRow, int excludeCol) {
		float det, sub[3][3];
		int x = 0, y = 0;
		for (int i = 0; i < 4; i++) {
			if (i == excludeRow) continue; // skip the excluded row
			y = 0;
			for (int j = 0; j < 4; j++) {
				if (j == excludeCol) continue; // skip the excluded column
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
		result = rotate(rotation)
			* translate(position);
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

#endif
;