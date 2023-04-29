#define NOMINMAX
#include <Windows.h>
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <random>
double PI = 3.14159265358979323846;
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
typedef struct {
	double x;
	double y;
	double z;
	double pitch;
	double yaw;
	double roll;
	double fov;
} cam;
cam camera = { 0,0,0,0,0,0, 300 }; //x,y,z,pitch,yaw,roll,fov
int randomInt(int min, int max) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(min, max);
	return dist(gen);
}
void setupConsole() {
	AllocConsole();
	FILE* stream;
	freopen_s(&stream, "CONOUT$", "w", stdout);
	freopen_s(&stream, "CONOUT$", "w", stderr);
	freopen_s(&stream, "CONIN$", "r", stdin);
	std::ios::sync_with_stdio();
}
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR pCmdLine, int nCmdShow) {
	setupConsole();
	const wchar_t CLASS_NAME[] = L"3d";
	WNDCLASS wc = {};
	wc.lpfnWndProc = WindowProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = CLASS_NAME;
	RegisterClass(&wc);

	// Create the window.
	HWND hwnd = CreateWindowEx(
		0,
		CLASS_NAME,
		L"3d",
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
		nullptr,
		nullptr,
		hInstance,
		nullptr);

	if (hwnd == nullptr) {
		return 0;
	}
	ShowWindow(hwnd, nCmdShow);
	MSG msg = {};
	while (GetMessage(&msg, nullptr, 0, 0)) {
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return 0;
}
COLORREF getcolor(std::string color) {
	if (color == "red") {
		return RGB(255, 0, 0);
	}
	else if (color == "green") {
		return RGB(0, 255, 0);
	}
	else if (color == "blue") {
		return RGB(0, 0, 255);
	}
	else if (color == "yellow") {
		return RGB(255, 255, 0);
	}
	else if (color == "purple") {
		return RGB(255, 0, 255);
	}
	else if (color == "cyan") {
		return RGB(0, 255, 255);
	}
	else if (color == "white") {
		return RGB(255, 255, 255);
	}
	else if (color == "black") {
		return RGB(0, 0, 0);
	}
	else {
		return RGB(255, 255, 255);
	}
}
void circle(HDC hdc, int x, int y, int radius, std::string color) {
	SelectObject(hdc, GetStockObject(DC_BRUSH));
	SetDCBrushColor(hdc, getcolor(color));
	SelectObject(hdc, CreatePen(PS_SOLID, 0, getcolor(color)));
	Ellipse(hdc, x - radius, y - radius, x + radius, y + radius);
	DeleteObject(SelectObject(hdc, GetStockObject(DC_PEN)));
}
void line(HDC hdc, int x1, int y1, int x2, int y2, int thickness, const std::string color) {
	COLORREF lineColor = getcolor(color);
	HPEN hPen = CreatePen(PS_SOLID, thickness, lineColor);
	HPEN hOldPen = static_cast<HPEN>(SelectObject(hdc, hPen));
	MoveToEx(hdc, x1, y1, nullptr);
	LineTo(hdc, x2, y2);
	HPEN hDotPen = CreatePen(PS_SOLID, 3, RGB(0, 255, 0));
	HPEN hOldDotPen = static_cast<HPEN>(SelectObject(hdc, hDotPen));

	SelectObject(hdc, hOldDotPen);
	DeleteObject(hDotPen);
	SelectObject(hdc, hOldPen);
	DeleteObject(hPen);
}
struct Vector3 {
	float x, y, z;
	Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
	Vector3 translate(float tx, float ty, float tz) const {
		return Vector3(x + tx, y + ty, z + tz);
	}
	Vector3 add(const Vector3& other) const { //add two vectors
		return Vector3(x + other.x, y + other.y, z + other.z);
	}
	Vector3 subtract(const Vector3& other) const {
		return Vector3(x - other.x, y - other.y, z - other.z);
	}
	float dotproduct(const Vector3& other) const {
		return x * other.x + y * other.y + z * other.z;
	}
	Vector3 scalarmult(float scalar) const { //multiplies a vector by a scalar
		return Vector3(x * scalar, y * scalar, z * scalar);
	}
	Vector3 crossproduct(const Vector3& other) const { //cross product of two vectors
		return Vector3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
	}
	float length() const {
		return sqrtf(x * x + y * y + z * z);
	}
	Vector3 normalize() const {
		float len = length();
		return Vector3(x / len, y / len, z / len);
	}
};
struct Vertex {
	Vector3 position;
	Vertex(float x, float y, float z) : position(x, y, z) {}
};
struct Vector2 {
	float x, y;
	Vector2(float x, float y) : x(x), y(y) {}
};
typedef struct Pyramid {
	std::vector<Vertex> vertices;
	Pyramid(float width, float depth, float height) {
		vertices = {
			Vertex(-width / 2.0f, 0.0f, -depth / 2.0f), // 0
			Vertex(width / 2.0f, 0.0f, -depth / 2.0f),  // 1
			Vertex(width / 2.0f, 0.0f, depth / 2.0f),   // 2
			Vertex(-width / 2.0f, 0.0f, depth / 2.0f),  // 3
			Vertex(0.0f, height, 0.0f) // 4
		};
	}
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
	static Matrix4 rotateX(float angle) {
		Matrix4 result;
		float radians = angle * (PI / 180.0f);
		result.m[1][1] = cosf(radians);
		result.m[1][2] = -sinf(radians);
		result.m[2][1] = sinf(radians);
		result.m[2][2] = cosf(radians);
		return result;
	}
	static Matrix4 rotateY(float angle) {
		Matrix4 result;
		float radians = angle * (PI / 180.0f);
		result.m[0][0] = cosf(radians);
		result.m[0][2] = sinf(radians);
		result.m[2][0] = -sinf(radians);
		result.m[2][2] = cosf(radians);
		return result;
	}
	static Matrix4 rotateZ(float angle) {
		Matrix4 result;
		float radians = angle * (PI / 180.0f);
		result.m[0][0] = cosf(radians);
		result.m[0][1] = -sinf(radians);
		result.m[1][0] = sinf(radians);
		result.m[1][1] = cosf(radians);
		return result;
	}
	Matrix4 transpose() const {
		Matrix4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = m[j][i];
			}
		}
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
	static Matrix4 perspective(float fov, float aspect_ratio, float near_clip, float far_clip) { //perspective projection matrix
		Matrix4 result;
		float f = 1.0f / tanf(fov * 0.5f * (PI / 180.0f));
		result.m[0][0] = f / aspect_ratio;
		result.m[1][1] = f;
		result.m[2][2] = (far_clip + near_clip) / (near_clip - far_clip);
		result.m[2][3] = (2.0f * far_clip * near_clip) / (near_clip - far_clip);
		result.m[3][2] = -1.0f;
		result.m[3][3] = 0.0f;
		return result;
	}
	static Matrix4 worldmatrix(float tx, float ty, float tz, float rx, float ry, float rz, float sx, float sy, float sz) { //t = translation, r = rotation, s = scale
		Matrix4 result = Matrix4::translate(tx, ty, tz).multiply(Matrix4::rotateX(rx)).multiply(Matrix4::rotateY(ry)).multiply(Matrix4::rotateZ(rz)).multiply(Matrix4::scale(sx, sy, sz));
		return result;
	}
	static Matrix4 viewmatrix(float camx, float camy, float camz) {
		Matrix4 result = Matrix4::rotateX(camx).multiply(Matrix4::rotateY(camy)).multiply(Matrix4::rotateZ(camz)).multiply(Matrix4::translate(-camx, -camy, -camz));
		return result;
	}
	static Vector3 projectVector(const Vector3& vectoroni, const Matrix4& world, const Matrix4& view, const Matrix4& projection) {
		Matrix4 result = world.multiply(view).multiply(projection);
		return result.vecmatrix(vectoroni);
	}
	static Vector2 project2D(const Vector3& vec3, float screenWidth, float screenHeight) { //converts a 3D vector to 2D to display
		float halfScreenWidth = screenWidth / 2;
		float halfScreenHeight = screenHeight / 2;
		float zNormalized = std::max(0.001f, vec3.z + 1.0f);
		float projectedX = vec3.x / zNormalized * halfScreenWidth + halfScreenWidth;
		float projectedY = vec3.y / zNormalized * halfScreenHeight + halfScreenHeight;
		return Vector2(projectedX, projectedY);
	}
	static Vector3 norm(const Vector3& vec3) { //normalizes a vector to NDC
		float x = (vec3.x + 1.0f) / 2.0f;
		float y = (vec3.y + 1.0f) / 2.0f;
		float z = (vec3.z + 1.0f) / 2.0f;
		return Vector3(x, y, z);
	}

};
std::vector<Vector3> extract(const Pyramid& pyramid) { //only works for pyramids
	std::vector<Vector3> extractedVertices;
	for (const Vertex& vertex : pyramid.vertices) {
		extractedVertices.push_back(vertex.position);
	}
	return extractedVertices;
}

Pyramid pyramid1(1000.0f, 1000.0f, 1000.0f);
Matrix4 world;
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	static std::tuple<double, double> mouse = { 0,0 }; //mouse x and y
	static std::vector<Vector2> projectedVertices;
	static double screenx, screeny;

	switch (uMsg) {
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	case WM_CREATE:
		SetTimer(hwnd, 1, 35, NULL);
		break;
	case WM_TIMER: {
		InvalidateRect(hwnd, NULL, TRUE);
		projectedVertices.clear(); //clears projected vertices so that they don't stack up
		std::vector<Vector3> extractedVertices = extract(pyramid1);
		Matrix4 worldMatrix = Matrix4::worldmatrix(0, 0, 0, 0, 0, 0, 1, 1, 1);
		Matrix4 viewMatrix = Matrix4::viewmatrix(camera.x, camera.y, camera.z)
			.multiply(Matrix4::rotateX(camera.pitch))
			.multiply(Matrix4::rotateY(camera.yaw))
			.multiply(Matrix4::rotateZ(camera.roll));
		Matrix4 projectionMatrix = Matrix4::perspective(camera.fov, screenx / screeny, 0.1, 30000000000.0);
		for (const auto& vertex : extractedVertices) {
			Vector3 projectedVec3 = Matrix4::projectVector(vertex, worldMatrix, viewMatrix, projectionMatrix);
			Vector3 normalizedVec3 = Matrix4::norm(projectedVec3);
			Vector2 projected = Matrix4::project2D(normalizedVec3, screenx, screeny);
			projectedVertices.push_back(projected);
		}

		break;
	}
	case WM_MOUSEMOVE: {
		std::get<0>(mouse) = LOWORD(lParam);
		std::get<1>(mouse) = HIWORD(lParam);
		break;
	}
	case WM_KEYDOWN: {
		float rotation_angle = 5.0f;
		float translation_amount = 1.0f;
		switch (wParam) {
		case VK_UP:
			camera.pitch += rotation_angle;
			break;
		case VK_DOWN:
			camera.pitch -= rotation_angle;
			break;
		case VK_LEFT:
			camera.yaw -= rotation_angle;
			break;
		case VK_RIGHT:
			camera.yaw += rotation_angle;
			break;
		case 'W':
			camera.z += translation_amount;
			break;
		case 'A':
			camera.x -= translation_amount;
			break;
		case 'S':
			camera.z -= translation_amount;
			break;
		case 'D':
			camera.x += translation_amount;
			break;
		case 'Q':
			camera.roll += rotation_angle;
			break;
		case 'E':
			camera.roll -= rotation_angle;
			break;
		}
		break;
	}
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);
		RECT rect;
		GetClientRect(hwnd, &rect);
		HDC memDC = CreateCompatibleDC(hdc); // Create a memory DC for double buffering
		HBITMAP memBM = CreateCompatibleBitmap(hdc, rect.right, rect.bottom);
		HGDIOBJ oldBM = SelectObject(memDC, memBM);
		FillRect(memDC, &rect, (HBRUSH)(COLOR_WINDOW + 1));
		screenx = rect.right - 100;
		screeny = rect.bottom - 100;
		if (projectedVertices.size() > 0) {
			for (size_t i = 1; i < projectedVertices.size(); i++) {
				line(memDC, projectedVertices[i - 1].x, projectedVertices[i - 1].y, projectedVertices[i].x, projectedVertices[i].y, 1, "black");
			}
			line(memDC, projectedVertices.back().x, projectedVertices.back().y, projectedVertices[0].x, projectedVertices[0].y, 1, "black");
		}
		BitBlt(hdc, 0, 0, rect.right, rect.bottom, memDC, 0, 0, SRCCOPY);
		SelectObject(memDC, oldBM);
		DeleteObject(memBM);
		DeleteDC(memDC); // delete the memory DC
		EndPaint(hwnd, &ps);
		break;
	}


	default:
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
}