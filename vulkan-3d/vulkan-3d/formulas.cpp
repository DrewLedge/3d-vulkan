#include <random>
#include <cmath>
#include <vector>
#include <ctime> //random seed based on time
#include <chrono> // random seed based on time
#include "forms.h"
int formulas::goodgen(int min, int max) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(min, max);
	return dist(gen);
};