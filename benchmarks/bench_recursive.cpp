#include "job_system/job_system.hpp"
#include "benchmarks/bench_common.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

using namespace job_system;

// Helper: standard merge operation
void Merge(int* data, size_t left, size_t mid, size_t right, int* temp) {
    size_t i = left;
    size_t j = mid;
    size_t k = left;

    while (i < mid && j < right) {
        if (data[i] <= data[j]) {
            temp[k++] = data[i++];
        } else {
            temp[k++] = data[j++];
        }
    }
    while (i < mid) temp[k++] = data[i++];
    while (j < right) temp[k++] = data[j++];

    for (size_t idx = left; idx < right; ++idx) {
        data[idx] = temp[idx];
    }
}

constexpr size_t SORT_THRESHOLD = 8192;

// 1. Sequential Recursive MergeSort
void SequentialMergeSort(int* data, size_t left, size_t right, int* temp) {
    if (right - left <= SORT_THRESHOLD) {
        std::sort(data + left, data + right);
        return;
    }

    size_t mid = left + (right - left) / 2;
    SequentialMergeSort(data, left, mid, temp);
    SequentialMergeSort(data, mid, right, temp);
    Merge(data, left, mid, right, temp);
}

// 2. OpenMP Task Recursive MergeSort
void OpenMPMergeSort(int* data, size_t left, size_t right, int* temp) {
    if (right - left <= SORT_THRESHOLD) {
        std::sort(data + left, data + right);
        return;
    }

    size_t mid = left + (right - left) / 2;

    #pragma omp task default(none) firstprivate(data, left, mid, temp)
    OpenMPMergeSort(data, left, mid, temp);

    #pragma omp task default(none) firstprivate(data, mid, right, temp)
    OpenMPMergeSort(data, mid, right, temp);

    #pragma omp taskwait
    Merge(data, left, mid, right, temp);
}

// 3. Job System 2.0 Recursive MergeSort
struct SortJobParams {
    int* data;
    size_t left;
    size_t right;
    int* temp;
    DefaultJobSystem* js;
};

void JobSystemMergeSortJob(Job* job, const void* rawData) {
    const auto* params = static_cast<const SortJobParams*>(rawData);
    int* data = params->data;
    size_t left = params->left;
    size_t right = params->right;
    int* temp = params->temp;
    auto* js = params->js;

    if (right - left <= SORT_THRESHOLD) {
        std::sort(data + left, data + right);
        return;
    }

    size_t mid = left + (right - left) / 2;

    // Allocate an intermediate sync job for merge
    // left and right children are spawned under the current job
    SortJobParams leftParams{data, left, mid, temp, js};
    Job* leftJob = js->CreateJobAsChild(job, &JobSystemMergeSortJob, leftParams);
    js->Run(leftJob);

    SortJobParams rightParams{data, mid, right, temp, js};
    Job* rightJob = js->CreateJobAsChild(job, &JobSystemMergeSortJob, rightParams);
    js->Run(rightJob);

    // Actively wait for both children while stealing/helping execute work!
    js->Wait(leftJob);
    js->Wait(rightJob);

    Merge(data, left, mid, right, temp);
}

void BenchmarkParallelSort(size_t arraySize, int iterations) {
    std::cout << ">>> Running Recursive Divide-and-Conquer Benchmark (Parallel MergeSort: "
              << arraySize << " ints) <<<\n";

    std::vector<int> masterData(arraySize);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < arraySize; ++i) {
        masterData[i] = static_cast<int>(rng() % 10000000);
    }

    std::vector<int> testData;
    std::vector<int> tempBuffer(arraySize);

    // 1. Sequential Baseline
    auto seqBench = bench::RunRepeated("Sequential MergeSort", iterations, 1.0, [&]() {
        testData = masterData;
        SequentialMergeSort(testData.data(), 0, arraySize, tempBuffer.data());
    });
    double seqMs = seqBench.medianMs;
    seqBench.speedupVsSeq = 1.0;

    // 2. OpenMP Task Baseline
    auto ompBench = bench::RunRepeated("OpenMP Task MergeSort", iterations, seqMs, [&]() {
        testData = masterData;
        #pragma omp parallel
        {
            #pragma omp single
            {
                OpenMPMergeSort(testData.data(), 0, arraySize, tempBuffer.data());
            }
        }
    });

    // 3. Job System 2.0 (Lock-Free Work Stealing)
    DefaultJobSystem js;
    auto jsBench = bench::RunRepeated("Job System 2.0 Work-Stealing", iterations, seqMs, [&]() {
        testData = masterData;
        SortJobParams params{testData.data(), 0, arraySize, tempBuffer.data(), &js};
        Job* root = js.CreateJob(&JobSystemMergeSortJob, params);
        js.Run(root);
        js.Wait(root);
    });

    std::vector<bench::BenchmarkResult> results = {seqBench, ompBench, jsBench};
    bench::PrintResultsTable("Parallel Recursive MergeSort (" + std::to_string(arraySize) + " ints)", results);
}

int main(int argc, char** argv) {
    size_t size = 5000000;
    if (argc > 1) {
        size = std::stoull(argv[1]);
    }
    BenchmarkParallelSort(size, 3);
    return 0;
}
