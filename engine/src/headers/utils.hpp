// A bunch of small utility functions for the project

#pragma once

#include <iostream>
#include <chrono>
#include <random>
#include <ctime>

#ifdef ENABLE_DEBUG
// get the file name based on a full file path
inline const char* getFileName(const char* path) {
    const char* file = strrchr(path, '/'); // last occurnce of '/'
    if (!file) { // if '/' doesnt exist, look for '//'
        file = strrchr(path, '\\');
    }
    // return the file name if it exists
    return file ? file + 1 : path;
}

void logwarning(const std::string& message, const char* file, int line) {
    std::cerr << "WARNING: " << message << "! (File: " << getFileName(file) << ", Line: " << line << ")\n";
}

void logwarning(const std::string& message, bool execute, const char* file, int line) {
    if (execute) {
        std::cerr << "WARNING: " << message << "! (File: " << getFileName(file) << ", Line: " << line << ")\n";
    }
}

#define LOG_WARNING(message) logwarning(message, __FILE__, __LINE__)
#define LOG_WARNING_IF(message, execute) logwarning(message, execute, __FILE__, __LINE__)

#else
#define LOG_WARNING(message) ((void)0)
#define LOG_WARNING_IF(condition, message) ((void)0)

#endif

using microseconds = std::chrono::microseconds;
using milliseconds = std::chrono::milliseconds;
using highResClock = std::chrono::high_resolution_clock;

namespace utils {
    void sep() {
        std::cout << "---------------------------------\n";
    }

    auto now() {
        return highResClock::now();
    }

    template<typename DurType>
    DurType duration(const std::chrono::time_point<highResClock>& start) {
        auto end = highResClock::now();
        return std::chrono::duration_cast<DurType>(end - start);
    }

    template<typename DurType>
    std::string durationString(const DurType& duration) {
        static_assert(std::is_same_v<DurType, microseconds> || std::is_same_v<DurType, milliseconds>, "Invalid duration type");
        std::string length = std::to_string(duration.count());

        if constexpr (std::is_same_v<DurType, microseconds>) {
            return length + " microseconds";
        }
        else if constexpr (std::is_same_v<DurType, milliseconds>) {
            return length + " milliseconds";
        }
    }

    template<typename DurType>
    void printDuration(const DurType& duration) {
        std::cout << "Time: " << durationString(duration) << "\n";;
    }
};
