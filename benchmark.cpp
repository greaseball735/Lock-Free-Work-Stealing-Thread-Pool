#include "job_system.hpp"
#include <iostream>
#include <chrono>

// Payload structure for matrix multiplication simulation
struct ParallelForData {
    float* array;
    uint32_t start;
    uint32_t end;
};

// Data processing kernel
void ProcessArrayData(Job*, const void* payload) {
    const ParallelForData* data = static_cast<const ParallelForData*>(payload);
    for (uint32_t i = data->start; i < data->end; ++i) {
        // Compute-bound workload
        data->array[i] = data->array[i] * 2.0f + 1.5f;
    }
}

// Recursive parallel split job
void ParallelForSplit(Job* job, const void* payload) {
    const ParallelForData* data = static_cast<const ParallelForData*>(payload);
    uint32_t count = data->end - data->start;

    if (count <= 1024) { // Minimum execution threshold
        ProcessArrayData(job, payload);
    } else {
        uint32_t mid = data->start + (count / 2);

        Job* left_job = JobSystem::CreateJobAsChild(job, ParallelForSplit);
        ParallelForData* left_data = reinterpret_cast<ParallelForData*>(left_job->data);
        left_data->array = data->array;
        left_data->start = data->start;
        left_data->end = mid;

        Job* right_job = JobSystem::CreateJobAsChild(job, ParallelForSplit);
        ParallelForData* right_data = reinterpret_cast<ParallelForData*>(right_job->data);
        right_data->array = data->array;
        right_data->start = mid;
        right_data->end = data->end;

        JobSystem::Run(left_job);
        JobSystem::Run(right_job);
    }
}

int main() {
    JobSystem::Initialize();

    constexpr uint32_t ELEMENT_COUNT = 16 * 1024 * 1024;
    float* data_array = new float[ELEMENT_COUNT];
    for(uint32_t i = 0; i < ELEMENT_COUNT; ++i) data_array[i] = 1.0f;

    // Benchmark Execution
    auto start_time = std::chrono::high_resolution_clock::now();

    Job* root_job = JobSystem::CreateJob([](Job*, const void*){});
    Job* parallel_job = JobSystem::CreateJobAsChild(root_job, ParallelForSplit);
    
    ParallelForData* p_data = reinterpret_cast<ParallelForData*>(parallel_job->data);
    p_data->array = data_array;
    p_data->start = 0;
    p_data->end = ELEMENT_COUNT;

    JobSystem::Run(parallel_job);
    JobSystem::Run(root_job);
    JobSystem::Wait(root_job);

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "Processed " << ELEMENT_COUNT << " elements via recursive Work-Stealing." << std::endl;
    std::cout << "Execution Time: " << duration << " ms" << std::endl;

    delete[] data_array;
    JobSystem::Shutdown();
    return 0;
}
