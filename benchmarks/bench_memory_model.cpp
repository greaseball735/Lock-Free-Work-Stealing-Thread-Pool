#include "job_system/job_system.hpp"
#include "benchmarks/bench_common.hpp"
#include <iostream>
#include <atomic>

using namespace job_system;

void RunMemoryModelExperiment(size_t taskCount, int iterations) {
    std::cout << ">>> Running Memory Model Ordering Experiment (" << taskCount << " tasks) <<<\n";
    std::cout << "Target: Linux x86_64 (TSO Hardware Memory Model)\n";
    std::cout << "Testing: Standard Weak Model vs x86 TSO Optimized vs Relaxed Experiment\n\n";

    // 1. Policy = Weak (Standard C++11 Acquire/Release + SeqCst Fences)
    JobSystem<4096, MemoryOrderingPolicy::Weak> jsWeak;
    std::atomic<uint64_t> counterWeak{0};
    auto benchWeak = bench::RunRepeated("Policy: Weak (Portable C++11)", iterations, 1.0, [&]() {
        counterWeak.store(0, std::memory_order_relaxed);
        Job* root = jsWeak.CreateJob([](Job*, const void*) {});
        for (size_t i = 0; i < taskCount; ++i) {
            Job* child = jsWeak.CreateJobAsChildLambda(root, [&counterWeak]() {
                counterWeak.fetch_add(1, std::memory_order_relaxed);
            });
            jsWeak.Run(child);
        }
        jsWeak.Run(root);
        jsWeak.Wait(root);
    });

    double baselineMs = benchWeak.medianMs;
    benchWeak.speedupVsSeq = 1.0; // Normalized baseline

    // 2. Policy = TSO_Optimized (Leveraging x86 TSO Store-Store and Load-Load guarantees)
    JobSystem<4096, MemoryOrderingPolicy::TSO_Optimized> jsTSO;
    std::atomic<uint64_t> counterTSO{0};
    auto benchTSO = bench::RunRepeated("Policy: x86 TSO Optimized", iterations, baselineMs, [&]() {
        counterTSO.store(0, std::memory_order_relaxed);
        Job* root = jsTSO.CreateJob([](Job*, const void*) {});
        for (size_t i = 0; i < taskCount; ++i) {
            Job* child = jsTSO.CreateJobAsChildLambda(root, [&counterTSO]() {
                counterTSO.fetch_add(1, std::memory_order_relaxed);
            });
            jsTSO.Run(child);
        }
        jsTSO.Run(root);
        jsTSO.Wait(root);
    });

    // 3. Policy = Relaxed_Experiment (Eliminates Store-Load fence on Pop)
    JobSystem<4096, MemoryOrderingPolicy::Relaxed_Experiment> jsRelaxed;
    std::atomic<uint64_t> counterRelaxed{0};
    auto benchRelaxed = bench::RunRepeated("Policy: Relaxed Experiment", iterations, baselineMs, [&]() {
        counterRelaxed.store(0, std::memory_order_relaxed);
        Job* root = jsRelaxed.CreateJob([](Job*, const void*) {});
        for (size_t i = 0; i < taskCount; ++i) {
            Job* child = jsRelaxed.CreateJobAsChildLambda(root, [&counterRelaxed]() {
                counterRelaxed.fetch_add(1, std::memory_order_relaxed);
            });
            jsRelaxed.Run(child);
        }
        jsRelaxed.Run(root);
        jsRelaxed.Wait(root);
    });

    std::vector<bench::BenchmarkResult> results = {benchWeak, benchTSO, benchRelaxed};
    bench::PrintResultsTable("Memory Model Comparison on x86 (" + std::to_string(taskCount) + " tasks)", results);
}

int main(int argc, char** argv) {
    size_t count = 200000;
    if (argc > 1) count = std::stoull(argv[1]);
    RunMemoryModelExperiment(count, 5);
    return 0;
}
