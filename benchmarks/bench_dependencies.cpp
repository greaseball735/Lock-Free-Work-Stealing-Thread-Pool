#include "job_system/job_system.hpp"
#include "benchmarks/bench_common.hpp"
#include <iostream>
#include <atomic>
#include <vector>

using namespace job_system;

/**
 * @brief Benchmark evaluating lock-free continuations (Part 5 of the blog series).
 *
 * Simulates a diamond dependency DAG:
 *           Stage A (Producer)
 *             /        \
 *     Stage B1          Stage B2 (Parallel Transformers)
 *             \        /
 *           Stage C (Consumer / Aggregator)
 *
 * Runs thousands of independent DAG instances in parallel.
 */

struct DagInstanceData {
    int inputVal{0};
    int b1Val{0};
    int b2Val{0};
    int outputVal{0};
    DefaultJobSystem* js{nullptr};
    std::atomic<uint64_t>* completedCount{nullptr};
};

void StageA([[maybe_unused]] Job* job, const void* raw) {
    auto* ctx = const_cast<DagInstanceData*>(reinterpret_cast<const DagInstanceData*>(raw));
    ctx->inputVal = 42;
    // Stage A completes; continuations (Stage B1 & Stage B2) are automatically fired
}

void StageB1([[maybe_unused]] Job* job, const void* raw) {
    auto* ctx = const_cast<DagInstanceData*>(reinterpret_cast<const DagInstanceData*>(raw));
    ctx->b1Val = ctx->inputVal * 2;
}

void StageB2([[maybe_unused]] Job* job, const void* raw) {
    auto* ctx = const_cast<DagInstanceData*>(reinterpret_cast<const DagInstanceData*>(raw));
    ctx->b2Val = ctx->inputVal + 10;
}

void StageC([[maybe_unused]] Job* job, const void* raw) {
    auto* ctx = const_cast<DagInstanceData*>(reinterpret_cast<const DagInstanceData*>(raw));
    ctx->outputVal = ctx->b1Val + ctx->b2Val;
    ctx->completedCount->fetch_add(1, std::memory_order_relaxed);
}

void BenchmarkDependencies(size_t dagCount, int iterations) {
    std::cout << ">>> Running Task Dependency DAG Benchmark (Part 5 Continuations: "
              << dagCount << " Diamond DAGs) <<<\n";

    // 1. Sequential execution
    std::vector<DagInstanceData> seqData(dagCount);
    std::atomic<uint64_t> seqCompleted{0};

    auto seqBench = bench::RunRepeated("Sequential Pipeline", iterations, 1.0, [&]() {
        seqCompleted.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < dagCount; ++i) {
            seqData[i].inputVal = 42;
            seqData[i].b1Val = seqData[i].inputVal * 2;
            seqData[i].b2Val = seqData[i].inputVal + 10;
            seqData[i].outputVal = seqData[i].b1Val + seqData[i].b2Val;
            seqCompleted.fetch_add(1, std::memory_order_relaxed);
        }
    });
    double seqMs = seqBench.medianMs;
    seqBench.speedupVsSeq = 1.0;

    // 2. Job System 2.0 with Continuations
    DefaultJobSystem js;
    std::vector<DagInstanceData> jsData(dagCount);
    std::atomic<uint64_t> jsCompleted{0};

    auto jsBench = bench::RunRepeated("Job System 2.0 Continuations", iterations, seqMs, [&]() {
        jsCompleted.store(0, std::memory_order_relaxed);
        Job* root = js.CreateJob([](Job*, const void*) {});

        for (size_t i = 0; i < dagCount; ++i) {
            jsData[i].completedCount = &jsCompleted;
            jsData[i].js = &js;

            // Create Stage C (runs after B1 and B2)
            Job* jobC = js.CreateJobAsChild(root, &StageC, jsData[i]);

            // Create Stage B1 and B2
            Job* jobB1 = js.CreateJob(&StageB1, jsData[i]);
            Job* jobB2 = js.CreateJob(&StageB2, jsData[i]);

            // Set up dependencies: B1 and B2 trigger C
            // Job C will be run as continuation of B1 and B2 via parent-child or chaining
            js.AddContinuation(jobB1, jobC);

            // Create Stage A
            Job* jobA = js.CreateJob(&StageA, jsData[i]);
            js.AddContinuation(jobA, jobB1);
            js.AddContinuation(jobA, jobB2);

            // Fire Stage A
            js.Run(jobA);
            js.Run(jobB2);
        }

        js.Run(root);
        js.Wait(root);
    });

    std::vector<bench::BenchmarkResult> results = {seqBench, jsBench};
    bench::PrintResultsTable("Lock-Free Continuations (" + std::to_string(dagCount) + " DAGs)", results);
}

int main(int argc, char** argv) {
    size_t count = 20000;
    if (argc > 1) count = std::stoull(argv[1]);
    BenchmarkDependencies(count, 5);
    return 0;
}
