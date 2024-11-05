// Drew's Math Library (DML)

#pragma once
#include <iostream>
#include <vector>

namespace mathc {
    constexpr float PI = 3.14159265359f;
    constexpr float DEG_TO_RAD = 0.01745329251f;
    constexpr float RAD_TO_DEG = 57.2957795131f;
    constexpr float EPSILON = 0.0000001f;
}

namespace dml {
    struct vec2;
    struct vec3;
    struct vec4;
    struct mat4;

    struct vec2 {
        float x, y;
        vec2(float x, float y) : x(x), y(y) {}

        friend std::ostream& operator<<(std::ostream& os, const vec2& v) {
            os << "(" << v.x << ", " << v.y << ")";
            return os;
        }

        bool operator==(const vec2& other) const {
            return std::abs(x - other.x) < mathc::EPSILON &&
                std::abs(y - other.y) < mathc::EPSILON;
        }

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

        float length() const {
            return std::sqrt(x * x + y * y);
        }

        vec2 operator/(float scalar) const {
            return vec2(x / scalar, y / scalar);
        }
    };

    struct alignas(16) vec3 {
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

        bool operator==(const vec3& other) const {
            return std::abs(x - other.x) < mathc::EPSILON &&
                std::abs(y - other.y) < mathc::EPSILON &&
                std::abs(z - other.z) < mathc::EPSILON;
        }

        friend std::ostream& operator<<(std::ostream& os, const vec3& v) {
            os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
            return os;
        }

        float length() const {
            return std::sqrt(x * x + y * y + z * z);
        }
    };

    struct vec4 {
        float x, y, z, w;

        vec4() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) {}
        vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
        vec4(const vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

        vec2 xy() const { return vec2(x, y); }
        vec3 xyz() const { return vec3(x, y, z); }

        vec4 conjugate() const {
            return vec4(-x, -y, -z, w);
        }

        bool operator==(const vec4& other) const {
            return std::abs(x - other.x) < mathc::EPSILON &&
                std::abs(y - other.y) < mathc::EPSILON &&
                std::abs(z - other.z) < mathc::EPSILON &&
                std::abs(w - other.w) < mathc::EPSILON;
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
            vec4 result;
            result.x = w * other.x + x * other.w + y * other.z - z * other.y;
            result.y = w * other.y - x * other.z + y * other.w + z * other.x;
            result.z = w * other.z + x * other.y - y * other.x + z * other.w;
            result.w = w * other.w - x * other.x - y * other.y - z * other.z;
            return result;
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

        float length() const {
            return std::sqrt(x * x + y * y + z * z + w * w);
        }
    };

    struct mat3 {
        union {
            struct {
                float m[3][3];
            };
            float flat[9];
        };

        mat3() {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    m[j][i] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        mat3(const mat3& other) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    m[j][i] = other.m[j][i];
                }
            }
        }

        mat3(float zero) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    m[j][i] = 0.0f;
                }
            }
        }
    };

    struct mat4 {
        union {
            struct {
                float m[4][4];
            };
            float flat[16];
        };

        mat4() {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    m[j][i] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        mat4(const mat4& other) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    m[j][i] = other.m[j][i];
                }
            }
        }

        mat4(float val) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    m[j][i] = val;
                }
            }
        }

        bool operator==(const mat4& other) const {
            bool equal = true;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    equal = equal && std::abs(m[j][i] - other.m[j][i]) < mathc::EPSILON;
                }
            }
            return equal;
        }

        mat4 operator*(const float val) {
            mat4 result;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    result.m[j][i] = m[j][i] * val;
                }
            }
            return result;
        }

        mat4 operator*(const mat4& other) const {
            mat4 result;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    result.m[j][i] = 0;
                    for (int k = 0; k < 4; k++) {
                        result.m[j][i] += m[k][i] * other.m[j][k];
                    }
                }
            }
            return result;
        }

        mat4& operator*=(const mat4& other) {
            mat4 temp;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    temp.m[j][i] = 0;
                    for (int k = 0; k < 4; k++) {
                        temp.m[j][i] += m[k][i] * other.m[j][k];
                    }
                }
            }
            *this = temp;
            return *this;
        }

        friend vec3 operator*(const mat4& mat, const vec3& vec) {
            float x = mat.m[0][0] * vec.x + mat.m[1][0] * vec.y + mat.m[2][0] * vec.z + mat.m[3][0];
            float y = mat.m[0][1] * vec.x + mat.m[1][1] * vec.y + mat.m[2][1] * vec.z + mat.m[3][1];
            float z = mat.m[0][2] * vec.x + mat.m[1][2] * vec.y + mat.m[2][2] * vec.z + mat.m[3][2];
            float w = mat.m[0][3] * vec.x + mat.m[1][3] * vec.y + mat.m[2][3] * vec.z + mat.m[3][3];
            if (w != 1.0f && w != 0.0f) {
                x /= w;
                y /= w;
                z /= w;
            }
            return vec3(x, y, z);
        }

        friend vec4 operator*(const mat4& mat, const vec4& vec) {
            float x = mat.m[0][0] * vec.x + mat.m[1][0] * vec.y + mat.m[2][0] * vec.z + mat.m[3][0] * vec.w;
            float y = mat.m[0][1] * vec.x + mat.m[1][1] * vec.y + mat.m[2][1] * vec.z + mat.m[3][1] * vec.w;
            float z = mat.m[0][2] * vec.x + mat.m[1][2] * vec.y + mat.m[2][2] * vec.z + mat.m[3][2] * vec.w;
            float w = mat.m[0][3] * vec.x + mat.m[1][3] * vec.y + mat.m[2][3] * vec.z + mat.m[3][3] * vec.w;
            return vec4(x, y, z, w);
        }

        mat4 operator+(const mat4& other) const {
            mat4 result;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    result.m[i][j] = m[i][j] + other.m[i][j];
                }
            }
            return result;
        }

        mat4& operator+=(const mat4& other) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    m[i][j] += other.m[i][j];
                }
            }
            return *this;
        }

        mat4 transpose() const {
            mat4 result;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    result.m[i][j] = m[j][i];
                }
            }
            return result;
        }
    };

    // ------------------ FORWARD DECLARATIONS ------------------ //
    mat4 lookAt(const vec3& eye, const vec3& f, const vec3& r, const vec3& u);
    mat4 lookAt(const vec3& eye, const vec3& target, vec3& upVec);
    mat4 rotateQuat(const vec4 q);

    // ------------------ VECTOR3 FORMULAS ------------------ //
    vec3 radians(const vec3& v) {
        return v * mathc::DEG_TO_RAD;
    }

    float radians(const float degree) {
        return degree * mathc::DEG_TO_RAD;
    }

    vec3 degrees(const vec3& v) {
        return v * mathc::RAD_TO_DEG;
    }

    float degrees(const float radian) {
        return radian * mathc::RAD_TO_DEG;
    }

    vec3 eulerToDir(const vec3& rotation) { // converts euler angle to direction vector
        // convert pitch and yaw from degrees to radians
        float pitch = radians(rotation.x);
        float yaw = radians(rotation.y);

        vec3 direction;
        direction.x = cos(yaw) * cos(pitch);
        direction.y = sin(pitch);
        direction.z = sin(yaw) * cos(pitch);
        return direction;
    }

    vec3 getForward(const vec3& rot) {
        float pitch = rot.x;
        float yaw = rot.y;

        vec3 forward;
        forward.x = -std::sin(yaw) * std::cos(pitch);
        forward.y = -std::sin(pitch);
        forward.z = std::cos(yaw) * std::cos(pitch);

        return forward;
    }

    vec3 cross(const vec3& a, const vec3& b) {
        vec3 res;
        res.x = a.y * b.z - a.z * b.y;
        res.y = a.z * b.x - a.x * b.z;
        res.z = a.x * b.y - a.y * b.x;

        return res;
    }

    vec3 getRight(const vec3& rot) {
        vec3 forward = getForward(rot);
        vec3 up(0.0f, -1.0f, 0.0f);
        return cross(forward, up);
    }

    vec3 getUp(const vec3& rot) {
        vec3 forward = getForward(rot);
        vec3 right = getRight(rot);
        return cross(right, forward);
    }

    float dot(const vec3& a, const vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    float dot(const vec4& a, const vec4& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    vec3 normalize(const vec3& v) {
        float length = v.length();

        if (length < mathc::EPSILON) {
            return vec3(0.0f, 0.0f, 0.0f);
        }

        return vec3(v.x / length, v.y / length, v.z / length);
    }

    vec3 quatToDir(const vec4& quat) {
        mat4 o = rotateQuat(quat).transpose();
        return o * vec3(0.0f, 0.0f, -1.0f);
    }

    // ------------------ VECTOR4 FORMULAS ------------------ //

    vec4 normalize(const vec4& v) {
        float length = v.length();
        if (length < mathc::EPSILON) {
            return vec4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        return vec4(v.x / length, v.y / length, v.z / length, v.w / length);
    }

    // function to turn a lookat matrix into a quaternion
    vec4 quatCast(const mat4& mat) {
        // calc the trace of the matrix, which is the sum of the diagonal elements
        // when the trace is positive, it has more numerical stability, and is more straightforward to calc
        float trace = mat.m[0][0] + mat.m[1][1] + mat.m[2][2];
        vec4 quaternion;

        if (trace > 0.0f) {
            float s = sqrt(trace + 1.0f);
            quaternion.w = s * 0.5f;
            s = 0.5f / s;

            // get the quaternion vector parts
            quaternion.x = (mat.m[2][1] - mat.m[1][2]) * s;
            quaternion.y = (mat.m[0][2] - mat.m[2][0]) * s;
            quaternion.z = (mat.m[1][0] - mat.m[0][1]) * s;
        }
        else {
            int i = 0;

            // find the greatest diagonal element to ensure better stability when computing square root
            if (mat.m[1][1] > mat.m[0][0]) i = 1;
            if (mat.m[2][2] > mat.m[i][i]) i = 2;

            int j = (i + 1) % 3; // the next diagonal element ater i (wraps around)
            int k = (i + 2) % 3; // the diagonal element after j (wraps around)

            float s = sqrt(mat.m[i][i] - mat.m[j][j] - mat.m[k][k] + 1.0f);
            quaternion[i] = s * 0.5f;
            s = 0.5f / s;

            // get the quaternion vector parts
            quaternion.w = (mat.m[k][j] - mat.m[j][k]) * s;
            quaternion[j] = (mat.m[j][i] + mat.m[i][j]) * s;
            quaternion[k] = (mat.m[k][i] + mat.m[i][k]) * s;
        }

        // return the calculated quaternion
        quaternion = normalize(quaternion);
        return quaternion;
    }

    vec4 targetToQuat(const vec3& position, const vec3& target) {
        vec3 up = { 0.0f, 1.0f, 0.0f };
        mat4 l = lookAt(target, position, up);
        return quatCast(l);
    }

    vec4 inverseQuat(const vec4& q) {
        float length = q.length();
        if (length < mathc::EPSILON) {
            return vec4(0.0f, 0.0f, 0.0f, 1.0f); // return identity quaternion
        }

        return vec4(-q.x / length, -q.y / length, -q.z / length, q.w / length);
    }

    vec4 angleAxis(float angle, const vec3& axis) {
        vec3 normAxis = normalize(axis);

        // compute the sin and cos of half the angle
        float half = angle * 0.5f;
        float sinHalf = sin(half);
        float cosHalf = cos(half);

        // create the quaternion
        return vec4(normAxis.x * sinHalf, normAxis.y * sinHalf, normAxis.z * sinHalf, cosHalf);
    }

    // ------------------ MATRIX4 FORMULAS ------------------ // 
    mat4 translate(const vec3 t) {
        mat4 result;
        result.m[3][0] = t.x;
        result.m[3][1] = t.y;
        result.m[3][2] = t.z;
        return result;
    }

    mat4 scale(const vec3 s) {
        mat4 result;
        result.m[0][0] = s.x;
        result.m[1][1] = s.y;
        result.m[2][2] = s.z;
        result.m[3][3] = 1.0f;
        return result;
    }

    mat4 rotate(const vec3 s) {
        mat4 result;
        float radX = radians(s.x);
        float radY = radians(s.y);
        float radZ = radians(s.z);

        mat4 rotX;
        rotX.m[1][1] = cosf(radX);
        rotX.m[2][1] = -sinf(radX);
        rotX.m[1][2] = sinf(radX);
        rotX.m[2][2] = cosf(radX);

        mat4 rotY;
        rotY.m[0][0] = cosf(radY);
        rotY.m[2][0] = sinf(radY);
        rotY.m[0][2] = -sinf(radY);
        rotY.m[2][2] = cosf(radY);

        mat4 rotZ;
        rotZ.m[0][0] = cosf(radZ);
        rotZ.m[1][0] = -sinf(radZ);
        rotZ.m[0][1] = sinf(radZ);
        rotZ.m[1][1] = cosf(radZ);

        result = rotZ * rotY * rotX;
        return result;
    }

    mat4 rotateQuat(const vec4 q) { // quaternian rotation
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

    mat4 viewMatrix(const vec3& position, const float& right, const float& up) {
        vec4 yRot = angleAxis(right, vec3(1.0f, 0.0f, 0.0f));
        vec4 xRot = angleAxis(up, vec3(0.0f, 1.0f, 0.0f));
        vec4 orientation = yRot * xRot;
        orientation = normalize(orientation);
        mat4 rotation = rotateQuat(orientation);

        mat4 translation = translate(position);
        return rotation * translation;
    }

    mat4 projection(float fov, float aspect, float nearPlane, float farPlane) {
        mat4 result(0);
        float fovRad = radians(fov);
        float tanHalf = tan(fovRad * 0.5f);

        result.m[0][0] = 1.0f / (aspect * tanHalf);
        result.m[1][1] = -1.0f / tanHalf;
        result.m[2][2] = farPlane / (nearPlane - farPlane);
        result.m[3][2] = (farPlane * nearPlane) / (nearPlane - farPlane);
        result.m[2][3] = -1.0f;

        return result;
    }

    mat3 mat4ToMat3(const mat4& m, int excludeCol, int excludeRow) {
        mat3 res;
        int x = 0, y = 0;
        for (int i = 0; i < 4; i++) {
            if (i == excludeRow) continue; // skip the excluded row
            y = 0;
            for (int j = 0; j < 4; j++) {
                if (j == excludeCol) continue; // skip the excluded column
                res.m[x][y] = m.m[j][i];
                y++;
            }
            x++;
        }
        return res;
    }

    float det3(const mat3& m) { // func to calc the determinant of a 3x3 matrix
        float d1 = m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2]);
        float d2 = m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[2][0] * m.m[1][2]);
        float d3 = m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[2][0] * m.m[1][1]);
        return d1 - d2 + d3;
    }

    float det4(const mat4& m) { // func to calc the determinant of a 4x4 matrix
        float res = 0.0f;
        for (int i = 0; i < 4; i++) {
            mat3 sub = mat4ToMat3(m, 0, i);
            int sign = (i % 2 == 0 ? 1 : -1);
            res += (sign * m.m[0][i] * det3(sub));
        }
        return res;
    }

    mat4 inverseMatrix(const mat4& m) {
        float d = det4(m); // get the determinant of the matrix

        // if the determinant is 0, its not invertible
        if (std::abs(d) < mathc::EPSILON) {
            std::cerr << "Matrix is not invertible!\n";
            return m;
        }

        // calculate the adjugate matrix
        // the adjugate is gotten from calculating the determinant of the submatrices and multiplying by the sign (which forms a checkerboard pattern)
        mat4 adjugate;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int sign = ((i + j) % 2 == 0) ? 1 : -1;
                mat3 sub = mat4ToMat3(m, j, i);

                // the adjugate is the tranpose of the cofactor (thus why its ij instead of ji)
                adjugate.m[i][j] = sign * det3(sub);
            }
        }

        // multiply the adjugate by the inversed determinant
        float invDet = 1.0f / d;
        return adjugate * invDet;
    }

    vec3 getCamWorldPos(const mat4& viewMat) {
        mat4 invView = inverseMatrix(viewMat);
        vec3 cameraWorldPosition(invView.m[3][0], invView.m[3][1], invView.m[3][2]);
        return cameraWorldPosition;
    }

    mat4 lookAt(const vec3& eye, const vec3& f, const vec3& r, const vec3& u) {
        mat4 result;

        result.m[0][0] = r.x;
        result.m[1][0] = r.y;
        result.m[2][0] = r.z;
        result.m[3][0] = dot(r * -1, eye);

        result.m[0][1] = u.x;
        result.m[1][1] = u.y;
        result.m[2][1] = u.z;
        result.m[3][1] = dot(u * -1, eye);

        // negate f due to right hand cord system
        result.m[0][2] = -f.x;
        result.m[1][2] = -f.y;
        result.m[2][2] = -f.z;
        result.m[3][2] = dot(f, eye);

        result.m[3][3] = 1.0f;
        return result;
    }

    mat4 lookAt(const vec3& eye, const vec3& target, vec3& upVec) {
        vec3 f = normalize((target - eye)); // forward vector
        vec3 r = normalize(cross(f, upVec)); // right vector
        vec3 u = normalize(cross(r, f)); // up vector
        return lookAt(eye, f, r, u);
    }

    mat4 gltfToMat4(const std::vector<double>& vec) {
        mat4 result;
        int index = 0;

        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                result.m[col][row] = static_cast<float>(vec[index]);
                index++;
            }
        }
        return result;
    }
};
