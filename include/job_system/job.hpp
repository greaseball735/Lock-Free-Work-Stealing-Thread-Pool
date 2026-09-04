#pragma once

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cstring>

namespace job_system {

struct Job;

// Job function signature: accepts pointer to the executing Job and pointer to in-place data
using JobFunction = void (*)(Job*, const void*);

/**
 * @brief Represents an individual unit of work in Job System 2.0.
 *
 * Sized and aligned to exactly 128 bytes (two 64-byte cache lines) to prevent
 * false sharing between worker threads. Contains inline storage for job data
 * (eliminating dynamic allocations) and an array of direct continuation jobs.
 */
struct alignas(128) Job {
    static constexpr size_t MAX_CONTINUATIONS = 7;
    static constexpr size_t MAX_JOB_DATA_SIZE = 48;

    JobFunction function{nullptr};
    Job* parent{nullptr};

    // Tracks unfinished subtasks. Initialized to 1 for the job itself.
    // Dynamically incremented when child jobs are added.
    // When this counter reaches 0, the job and all its children are complete.
    std::atomic<int32_t> unfinishedJobs{0};

    // Number of continuations registered via AddContinuation
    std::atomic<int32_t> continuationCount{0};

    // Continuations spawned automatically upon completion
    Job* continuations[MAX_CONTINUATIONS]{nullptr};

    // In-place storage for job arguments (avoids separate heap allocations)
    alignas(8) char data[MAX_JOB_DATA_SIZE]{0};

    // Helper to safely access typed inline data
    template <typename T>
    const T* GetData() const {
        static_assert(sizeof(T) <= MAX_JOB_DATA_SIZE, "Requested type exceeds in-place Job data size");
        return reinterpret_cast<const T*>(data);
    }

    template <typename T>
    T* GetData() {
        static_assert(sizeof(T) <= MAX_JOB_DATA_SIZE, "Requested type exceeds in-place Job data size");
        return reinterpret_cast<T*>(data);
    }
};

static_assert(sizeof(Job) == 128, "Job struct must be exactly 128 bytes to prevent false sharing and fit cache lines");
static_assert(alignof(Job) == 128, "Job struct must be aligned to 128 bytes");

} // namespace job_system
