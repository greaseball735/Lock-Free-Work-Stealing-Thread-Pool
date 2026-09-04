#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iomanip>
#include <algorithm>

namespace bench {

class Timer {
public:
    Timer() : m_start(std::chrono::high_resolution_clock::now()) {}

    void Reset() {
        m_start = std::chrono::high_resolution_clock::now();
    }

    double ElapsedMs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - m_start).count();
    }

    double ElapsedSeconds() const {
        return ElapsedMs() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

struct BenchmarkResult {
    std::string name;
    double medianMs;
    double minMs;
    double maxMs;
    double speedupVsSeq;
};

inline void PrintResultsTable(const std::string& title, const std::vector<BenchmarkResult>& results) {
    std::cout << "\n========================================================================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(32) << "Implementation"
              << std::right << std::setw(14) << "Median (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)"
              << std::setw(16) << "Speedup vs Seq"
              << "\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(32) << r.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(14) << r.medianMs
                  << std::setw(12) << r.minMs
                  << std::setw(12) << r.maxMs
                  << std::setw(14) << r.speedupVsSeq << "x"
                  << "\n";
    }
    std::cout << "========================================================================================\n\n";
}

template <typename Func>
BenchmarkResult RunRepeated(const std::string& name, int iterations, double seqBaselineMs, Func&& func) {
    std::vector<double> timings;
    timings.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        Timer t;
        func();
        timings.push_back(t.ElapsedMs());
    }

    std::sort(timings.begin(), timings.end());
    double median = timings[timings.size() / 2];
    double minVal = timings.front();
    double maxVal = timings.back();
    double speedup = (median > 0.0) ? (seqBaselineMs / median) : 1.0;

    return BenchmarkResult{name, median, minVal, maxVal, speedup};
}

} // namespace bench
