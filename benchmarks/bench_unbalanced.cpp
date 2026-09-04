#include "job_system/job_system.hpp"
#include "job_system/parallel_for.hpp"
#include "baselines/mutex_thread_pool.hpp"
#include "benchmarks/bench_common.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <omp.h>

using namespace job_system;
using namespace baselines;

// Computes Mandelbrot iterations for pixel (px, py)
inline int MandelbrotPixel(int px, int py, int width, int height, int maxIter) {
    float x0 = (static_cast<float>(px) - static_cast<float>(width) / 2.0f) * 4.0f / static_cast<float>(width);
    float y0 = (static_cast<float>(py) - static_cast<float>(height) / 2.0f) * 4.0f / static_cast<float>(height);

    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;

    while (x * x + y * y <= 4.0f && iter < maxIter) {
        float xTemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xTemp;
        iter++;
    }
    return iter;
}

void BenchmarkUnbalancedMandelbrot(int width, int height, int maxIter, int iterations) {
    std::cout << ">>> Running Irregular/Unbalanced Workload: Mandelbrot ("
              << width << "x" << height << ", max " << maxIter << " iter) <<<\n";

    std::vector<int> imageSeq(width * height, 0);
    std::vector<int> imageMutex(width * height, 0);
    std::vector<int> imageOmpStatic(width * height, 0);
    std::vector<int> imageOmpDynamic(width * height, 0);
    std::vector<int> imageJobSystem(width * height, 0);

    // 1. Sequential Baseline
    auto seqBench = bench::RunRepeated("Sequential", iterations, 1.0, [&]() {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                imageSeq[y * width + x] = MandelbrotPixel(x, y, width, height, maxIter);
            }
        }
    });
    double seqMs = seqBench.medianMs;
    seqBench.speedupVsSeq = 1.0;

    // 2. OpenMP Static Scheduling (Suffers from heavy load imbalance)
    auto ompStaticBench = bench::RunRepeated("OpenMP static schedule", iterations, seqMs, [&]() {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                imageOmpStatic[y * width + x] = MandelbrotPixel(x, y, width, height, maxIter);
            }
        }
    });

    // 3. OpenMP Dynamic Scheduling (Central queue contention)
    auto ompDynamicBench = bench::RunRepeated("OpenMP dynamic schedule", iterations, seqMs, [&]() {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                imageOmpDynamic[y * width + x] = MandelbrotPixel(x, y, width, height, maxIter);
            }
        }
    });

    // 4. Mutex Thread Pool (Row-by-row chunks)
    MutexThreadPool mutexPool;
    auto mutexBench = bench::RunRepeated("Mutex Thread Pool", iterations, seqMs, [&]() {
        for (int y = 0; y < height; ++y) {
            mutexPool.Enqueue([&imageMutex, y, width, height, maxIter]() {
                for (int x = 0; x < width; ++x) {
                    imageMutex[y * width + x] = MandelbrotPixel(x, y, width, height, maxIter);
                }
            });
        }
        mutexPool.WaitAll();
    });

    // 5. Job System 2.0 (Lock-Free Work Stealing)
    DefaultJobSystem js;
    struct MandelContext {
        int* output;
        int width;
        int height;
        int maxIter;
    };
    MandelContext ctx{imageJobSystem.data(), width, height, maxIter};

    auto jsBench = bench::RunRepeated("Job System 2.0 Work-Stealing", iterations, seqMs, [&]() {
        parallel_for_index(js, 0, height, [](size_t rowStart, size_t rowCount, const void* userCtx) {
            const auto* c = static_cast<const MandelContext*>(userCtx);
            for (size_t y = rowStart; y < rowStart + rowCount; ++y) {
                for (int x = 0; x < c->width; ++x) {
                    c->output[y * c->width + x] = MandelbrotPixel(x, static_cast<int>(y), c->width, c->height, c->maxIter);
                }
            }
        }, &ctx, CountSplitter(16));
    });

    std::vector<bench::BenchmarkResult> results = {
        seqBench, ompStaticBench, ompDynamicBench, mutexBench, jsBench
    };
    bench::PrintResultsTable("Irregular Workload: Mandelbrot Load Balancing (" + std::to_string(width) + "x" + std::to_string(height) + ")", results);
}

int main(int argc, char** argv) {
    int dim = 2048;
    int maxIter = 500;
    if (argc > 1) dim = std::stoi(argv[1]);
    if (argc > 2) maxIter = std::stoi(argv[2]);

    BenchmarkUnbalancedMandelbrot(dim, dim, maxIter, 3);
    return 0;
}
