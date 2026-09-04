#pragma once

#include "job.hpp"
#include <cstdint>
#include <cstddef>

namespace job_system {

/**
 * @brief Thread-local ring buffer allocator for Job instances (Part 2 of blog series).
 *
 * Traditional new/delete incurs heavy mutex contention inside malloc/free and cache
 * thrashing. This specialized allocator eliminates heap allocations entirely:
 * - Each thread gets its own contiguous ring-buffer of preallocated Job structs.
 * - Allocations require ZERO locks, ZERO atomic operations, and ZERO syscalls.
 * - Jobs do not need to be individually deleted/freed during execution; their slots
 *   are naturally recycled as the thread-local ring buffer wraps around.
 */
template <size_t MaxJobs = 4096>
class ThreadLocalJobAllocator {
    static_assert((MaxJobs & (MaxJobs - 1)) == 0, "MaxJobs must be a power of two");

public:
    static constexpr size_t CAPACITY = MaxJobs;
    static constexpr size_t MASK = MaxJobs - 1;

    static Job* Allocate() {
        // Fast thread-local increment without atomics
        const uint32_t index = t_allocatedJobs++;
        Job* job = &t_jobPool[index & MASK];

        // Reset state for newly recycled job
        job->function = nullptr;
        job->parent = nullptr;
        job->unfinishedJobs.store(1, std::memory_order_relaxed);
        job->continuationCount.store(0, std::memory_order_relaxed);
        return job;
    }

    static uint32_t GetAllocatedCount() {
        return t_allocatedJobs;
    }

private:
    static thread_local Job t_jobPool[CAPACITY];
    static thread_local uint32_t t_allocatedJobs;
};

template <size_t MaxJobs>
thread_local Job ThreadLocalJobAllocator<MaxJobs>::t_jobPool[ThreadLocalJobAllocator<MaxJobs>::CAPACITY];

template <size_t MaxJobs>
thread_local uint32_t ThreadLocalJobAllocator<MaxJobs>::t_allocatedJobs = 0;

using DefaultJobAllocator = ThreadLocalJobAllocator<4096>;

inline Job* AllocateJob() {
    return DefaultJobAllocator::Allocate();
}

} // namespace job_system
