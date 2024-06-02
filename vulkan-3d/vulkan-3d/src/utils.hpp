// A bunch of small utility functions for the project

#pragma once

#include <iostream>
#include <chrono>
#include <random>
#include <ctime>

using microseconds = std::chrono::microseconds;
using milliseconds = std::chrono::milliseconds;
using highResClock = std::chrono::high_resolution_clock;

class utils {
public:
	static void sep() {
		std::cout << "---------------------------------" << std::endl;
	}

	static auto now() {
		return highResClock::now();
	}

	template<typename DurType>
	static DurType duration(const std::chrono::time_point<highResClock>& start) {
		auto end = highResClock::now();
		return std::chrono::duration_cast<DurType>(end - start);
	}

	template<typename DurType>
	static std::string durationString(const DurType& duration) {
		std::string length = std::to_string(duration.count());

		if constexpr (std::is_same_v<DurType, microseconds>) {
			return length + " microseconds";
		}
		else if constexpr (std::is_same_v<DurType, milliseconds>) {
			return length + " milliseconds";
		}
		else {
			static_assert(std::is_same_v<DurType, microseconds> || std::is_same_v<DurType, milliseconds>, "Invalid duration type");
		}
	}

	template<typename DurType>
	static void printDuration(const DurType& duration) {
		std::cout << "Time: " << durationString(duration) << std::endl;;
	}

	static int random(int min, int max) {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dist(min, max);
		return dist(gen);
	};
};