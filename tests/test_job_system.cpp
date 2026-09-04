#include "job_system/job_system.hpp"
#include "job_system/parallel_for.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <numeric>

using namespace job_system;

void test_basic_job() {
    std::cout << "[TEST] Running test_basic_job..." << std::endl;
    DefaultJobSystem js;

    struct TestData {
        int a;
        int b;
        int result;
    };

    TestData data{10, 20, 0};

    Job* job = js.CreateJob([](Job*, const void* raw) {
        auto* d = const_cast<TestData*>(reinterpret_cast<const TestData*>(raw));
        d->result = d->a + d->b;
    }, data);

    js.Run(job);
    js.Wait(job);

    // Fetch result from the job's inline data copy
    const auto* resultData = job->GetData<TestData>();
    assert(resultData->result == 30);
    std::cout << "[PASS] Basic job executed successfully. Result: " << resultData->result << std::endl;
}

void test_dynamic_parallelism_children() {
    std::cout << "[TEST] Running test_dynamic_parallelism_children..." << std::endl;
    DefaultJobSystem js;

    constexpr int NUM_CHILDREN = 500;
    std::atomic<int> counter{0};

    Job* root = js.CreateJob([](Job*, const void*) {
        // Root job payload
    });

    for (int i = 0; i < NUM_CHILDREN; ++i) {
        Job* child = js.CreateJobAsChildLambda(root, [&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
        js.Run(child);
    }

    js.Run(root);
    js.Wait(root);

    assert(counter.load() == NUM_CHILDREN);
    std::cout << "[PASS] Dynamic child jobs completed. Counter: " << counter.load() << std::endl;
}

void test_parallel_for() {
    std::cout << "[TEST] Running test_parallel_for..." << std::endl;
    DefaultJobSystem js;

    constexpr size_t N = 100000;
    std::vector<int> data(N, 0);

    // Parallel fill using CountSplitter
    parallel_for_each(js, N, [&data](size_t i) {
        data[i] = static_cast<int>(i * 2);
    }, CountSplitter(1024));

    for (size_t i = 0; i < N; ++i) {
        assert(data[i] == static_cast<int>(i * 2));
    }
    std::cout << "[PASS] parallel_for_each verified across " << N << " items." << std::endl;
}

void test_continuations() {
    std::cout << "[TEST] Running test_continuations (Part 5)..." << std::endl;
    DefaultJobSystem js;

    std::atomic<int> stage1{0};
    std::atomic<int> stage2{0};

    // Stage 1 job
    Job* first = js.CreateJobLambda([&stage1]() {
        stage1.store(1, std::memory_order_release);
    });

    // Continuation: must only run AFTER first job has completed
    Job* second = js.CreateJobLambda([&stage1, &stage2]() {
        assert(stage1.load(std::memory_order_acquire) == 1 && "Continuation executed prematurely!");
        stage2.store(2, std::memory_order_release);
    });

    js.AddContinuation(first, second);

    // Only run first! The job system will automatically fire second upon first's completion.
    js.Run(first);
    js.Wait(second);

    assert(stage1.load() == 1);
    assert(stage2.load() == 2);
    std::cout << "[PASS] Continuations executed in correct dependency order." << std::endl;
}

int main() {
    std::cout << "=== Job System 2.0 Verification Test Suite ===" << std::endl;
    test_basic_job();
    test_dynamic_parallelism_children();
    test_parallel_for();
    test_continuations();
    std::cout << "All unit tests passed successfully!" << std::endl;
    return 0;
}
