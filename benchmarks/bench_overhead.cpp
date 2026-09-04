#include "job_system/job_system.hpp"
#include "baselines/mutex_thread_pool.hpp"
#include "benchmarks/bench_common.hpp"
#include <iostream>
#include <atomic>
#include <omp.h>

using namespace job_system;
using namespace baselines;

void BenchmarkOverhead(size_t taskCount, int iterations) {
    std::cout << ">>> Running Overhead / Fine-Grained Task Benchmark (" << taskCount << " tasks) <<<\n";

    // 1. Sequential Baseline
    std::atomic<uint64_t> seqCounter{0};
    auto seqBench = bench::RunRepeated("Sequential", iterations, 1.0, [&]() {
        seqCounter.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < taskCount; ++i) {
            seqCounter.fetch_add(1, std::memory_order_relaxed);
        }
    });

    double seqMs = seqBench.medianMs;
    seqBench.speedupVsSeq = 1.0;

    // 2. Mutex Thread Pool
    MutexThreadPool mutexPool;
    std::atomic<uint64_t> mutexCounter{0};
    auto mutexBench = bench::RunRepeated("Mutex Thread Pool", iterations, seqMs, [&]() {
        mutexCounter.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < taskCount; ++i) {
            mutexPool.Enqueue([&mutexCounter]() {
                mutexCounter.fetch_add(1, std::memory_order_relaxed);
            });
        }
        mutexPool.WaitAll();
    });

    // 3. OpenMP Task Baseline
    std::atomic<uint64_t> ompCounter{0};
    auto ompBench = bench::RunRepeated("OpenMP Tasks", iterations, seqMs, [&]() {
        ompCounter.store(0, std::memory_order_relaxed);
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (size_t i = 0; i < taskCount; ++i) {
                    #pragma omp task
                    {
                        ompCounter.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    });

    // 4. Job System 2.0 (Molecular Matters Lock-Free Work-Stealing)
    DefaultJobSystem js;
    std::atomic<uint64_t> jsCounter{0};
    auto jsBench = bench::RunRepeated("Job System 2.0 (Lock-Free)", iterations, seqMs, [&]() {
        jsCounter.store(0, std::memory_order_relaxed);
        Job* root = js.CreateJob([](Job*, const void*) {});

        for (size_t i = 0; i < taskCount; ++i) {
            Job* child = js.CreateJobAsChildLambda(root, [&jsCounter]() {
                jsCounter.fetch_add(1, std::memory_order_relaxed);
            });
            js.Run(child);
        }

        js.Run(root);
        js.Wait(root);
    });

    std::vector<bench::BenchmarkResult> results = {seqBench, mutexBench, ompBench, jsBench};
    bench::PrintResultsTable("Task Overhead & Scheduling Latency (" + std::to_string(taskCount) + " micro-tasks)", results);
}

int main(int argc, char** argv) {
    size_t count = 100000;
    if (argc > 1) {
        count = std::stoull(argv[1]);
    }
    BenchmarkOverhead(count, 5);
    return 0;
}
