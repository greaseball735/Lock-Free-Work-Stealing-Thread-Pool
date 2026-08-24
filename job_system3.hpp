#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

// -----------------------------------------------------------------------------
// Job Descriptor (64-byte aligned to prevent false sharing)
// -----------------------------------------------------------------------------
struct alignas(64) Job {
    using JobFunction = void(*)(Job*, const void*);

    JobFunction function;
    Job* parent;
    std::atomic<int32_t> unfinished_jobs;

    // In-place storage for small payload data to fit inside a 64-byte cache line
    static constexpr size_t PAYLOAD_SIZE = 64 - sizeof(JobFunction) - sizeof(Job*) - sizeof(std::atomic<int32_t>);
    uint8_t data[PAYLOAD_SIZE];
};

// -----------------------------------------------------------------------------
// Lock-Free Chase-Lev Work-Stealing Deque
// -----------------------------------------------------------------------------
class WorkStealingQueue {
private:
// Increase from 8192 to 65536 to prevent ring-buffer wrap-around corruption
// during deep recursive DAG generations.
static constexpr int32_t CAPACITY = 65536; 
// static constexpr uint32_t MAX_JOBS_PER_THREAD = 65536;
    // static constexpr int32_t CAPACITY = 4096;
    // static constexpr int32_t MASK = CAPACITY - 1;
// Increase ceiling to safely house massive task graphs
static constexpr uint32_t MAX_JOBS_PER_THREAD = 262144; 


    alignas(64) std::atomic<int32_t> top{0};
    alignas(64) std::atomic<int32_t> bottom{0};
    Job* ring_buffer[CAPACITY];

public:
    void Push(Job* job);
    Job* Pop();
    Job* Steal();
};

// -----------------------------------------------------------------------------
// Core API
// -----------------------------------------------------------------------------
namespace JobSystem {
    void Initialize();
    void Shutdown();

    Job* AllocateJob();
    
    // Core structural API
    Job* CreateJob(Job::JobFunction function);
    Job* CreateJobAsChild(Job* parent, Job::JobFunction function);
    
    void Run(Job* job);
    void Wait(const Job* job);
    bool HasJobCompleted(const Job* job);
    void Execute(Job* job);

    // Dynamic Parallelism API
    template<typename T>
    void ParallelFor(T* data, uint32_t count, void(*function)(T*));
}
