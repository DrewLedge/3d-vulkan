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
cam camera = { 0,30,0,0,0,0,90 }; //x,y,z,pitch,yaw,roll,fov
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

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	static std::vector<std::tuple<double, double>>polys;
	static std::vector<std::tuple<double, double>>lineseg;
	static std::tuple<int, int, int, int> mouse = { 0,0,0,0 };
	static bool mousedown = false;
	static bool on_ground = false;

	switch (uMsg) {
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	case WM_CREATE:
		SetTimer(hwnd, 1, 15, NULL);
		break;
	case WM_TIMER: {
		InvalidateRect(hwnd, NULL, TRUE);
	}
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);
		RECT rect;
		GetClientRect(hwnd, &rect);
		HDC memDC = CreateCompatibleDC(hdc); //Create a memory DC for double buffering
		HBITMAP memBM = CreateCompatibleBitmap(hdc, rect.right, rect.bottom);
		HGDIOBJ oldBM = SelectObject(memDC, memBM);
		FillRect(memDC, &rect, (HBRUSH)(COLOR_WINDOW + 1));
		circle(memDC, 500, 500, 20, "red");
		line(memDC, 1000, 100, 1400, 900, 60, "black");
		BitBlt(hdc, 0, 0, rect.right, rect.bottom, memDC, 0, 0, SRCCOPY);
		SelectObject(memDC, oldBM);
		DeleteObject(memBM);
		DeleteDC(memDC); //delete the memory DC

		EndPaint(hwnd, &ps);
		break;
	}

	default:
		return DefWindowProc(hwnd, uMsg, wParam, lParam);
	}
}