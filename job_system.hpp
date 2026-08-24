#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

// ============================================================================
// Configuration
// ============================================================================

namespace JobSystemConfig {

    // Must be a power of two because the deque uses index & MASK.
    static constexpr uint64_t QUEUE_CAPACITY = 1ull << 16; // 65536

    // Maximum number of jobs allocated by one thread before its allocator
    // is exhausted.
    //
    // 65536 jobs * sizeof(Job) is still reasonably small, but note that
    // this storage lives in thread-local memory.
    static constexpr uint32_t MAX_JOBS_PER_THREAD = 65536;

    // Number of failed scheduling attempts before yielding.
    static constexpr uint32_t SPIN_BEFORE_YIELD = 256;

    // Default grain size for ParallelFor.
    static constexpr uint32_t DEFAULT_PARALLEL_FOR_GRAIN = 256;

    // CPU affinity is deliberately optional. The default is false because
    // blindly mapping thread_id -> CPU is not correct on all machines.
    static constexpr bool PIN_WORKERS = false;

} // namespace JobSystemConfig


// ============================================================================
// Job
// ============================================================================
//
// The descriptor is intended to fit in one cache line on the usual
// 64-bit platforms.
//
// Payload is stored directly inside the descriptor, avoiding an additional
// heap allocation for small task arguments.
//
// ============================================================================

struct alignas(64) Job {

    using JobFunction = void(*)(Job* job, const void* data);

    JobFunction function = nullptr;
    Job* parent = nullptr;
    std::atomic<int32_t> unfinished_jobs{0};

    // With the fields above this leaves enough inline space for small
    // task payloads on normal 64-bit platforms.
    static constexpr std::size_t PAYLOAD_SIZE =
        64 -
        sizeof(JobFunction) -
        sizeof(Job*) -
        sizeof(std::atomic<int32_t>);

    static_assert(PAYLOAD_SIZE >= 32,
                  "Job payload is unexpectedly small.");

    alignas(8) std::uint8_t data[PAYLOAD_SIZE];

    Job() noexcept {
        // We deliberately don't clear the payload here.
        // CreateJob(payload) copies the required bytes into it.
    }

    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;
};


// ============================================================================
// Lock-Free Chase-Lev Work-Stealing Queue
// ============================================================================
//
// One owner thread:
//     Push()
//     Pop()
//
// Multiple remote threads:
//     Steal()
//
// Local execution is LIFO:
//     Pop() operates on bottom.
//
// Stealing is FIFO:
//     Steal() operates on top.
//
// ============================================================================

class WorkStealingQueue {
private:
    static constexpr std::uint64_t CAPACITY =
        JobSystemConfig::QUEUE_CAPACITY;

    static constexpr std::uint64_t MASK = CAPACITY - 1;

    static_assert((CAPACITY & MASK) == 0,
                  "QUEUE_CAPACITY must be a power of two.");

    // Separate top and bottom onto independent cache lines.
    alignas(64) std::atomic<std::int64_t> top{0};
    alignas(64) std::atomic<std::int64_t> bottom{0};

    // The queue itself is bounded.
    Job* ring_buffer[CAPACITY];

public:
    WorkStealingQueue() noexcept = default;

    WorkStealingQueue(const WorkStealingQueue&) = delete;
    WorkStealingQueue& operator=(const WorkStealingQueue&) = delete;

    // Called only by the owner thread.
    void Push(Job* job) noexcept;

    // Called only by the owner thread.
    Job* Pop() noexcept;

    // Called concurrently by stealing threads.
    Job* Steal() noexcept;

    std::int64_t ApproximateSize() const noexcept;
};


// ============================================================================
// JobSystem API
// ============================================================================

namespace JobSystem {

    // worker_count:
    //     0 -> use std::thread::hardware_concurrency()
    //     N -> use N total execution threads, including the calling/main thread.
    //
    // Thus Initialize(8) creates:
    //     main thread + 7 worker threads.
    void Initialize(std::uint32_t worker_count = 0);

    void Shutdown();

    bool IsInitialized() noexcept;

    std::uint32_t ThreadCount() noexcept;

    // ------------------------------------------------------------------------
    // Job allocation
    // ------------------------------------------------------------------------

    Job* AllocateJob();

    // Basic job with no payload.
    Job* CreateJob(Job::JobFunction function);

    // Create a job and copy a trivially-copyable payload into Job::data.
    template <typename T>
    Job* CreateJob(Job::JobFunction function, const T& payload);

    // Child job automatically contributes to parent's unfinished counter.
    Job* CreateJobAsChild(Job* parent, Job::JobFunction function);

    template <typename T>
    Job* CreateJobAsChild(
        Job* parent,
        Job::JobFunction function,
        const T& payload
    );

    // ------------------------------------------------------------------------
    // Scheduling / execution
    // ------------------------------------------------------------------------

    void Run(Job* job);

    void Execute(Job* job);

    // ------------------------------------------------------------------------
    // Completion
    // ------------------------------------------------------------------------

    bool HasJobCompleted(const Job* job);

    // Cooperative wait:
    // while waiting, the caller executes other available jobs.
    void Wait(const Job* job);

    // ------------------------------------------------------------------------
    // ParallelFor
    // ------------------------------------------------------------------------

    template <typename T>
    using ParallelForFunction = void(*)(T*);

    template <typename T>
    void ParallelFor(
        T* data,
        std::uint32_t count,
        ParallelForFunction<T> function,
        std::uint32_t grain = JobSystemConfig::DEFAULT_PARALLEL_FOR_GRAIN
    );

} // namespace JobSystem


// ============================================================================
// Template implementation
// ============================================================================

namespace JobSystem {

    namespace detail {

        template <typename T>
        struct ParallelForTask {

            T* data;

            std::uint32_t begin;
            std::uint32_t end;

            ParallelForFunction<T> function;

            std::uint32_t grain;
        };

        

        template <typename T>
        void ParallelForTaskFunction(
            Job* job,
            const void* raw_data
        ) {
            const auto& task =
                *static_cast<const ParallelForTask<T>*>(raw_data);

            const std::uint32_t begin = task.begin;
            const std::uint32_t end   = task.end;

            if (begin >= end) {
                return;
            }

            const std::uint32_t count = end - begin;

            // Small enough: execute directly.
            if (count <= task.grain) {
                for (std::uint32_t i = begin; i < end; ++i) {
                    task.function(&task.data[i]);
                }

                return;
            }

            // Split recursively.
            const std::uint32_t middle =
                begin + count / 2;

            ParallelForTask<T> left{
                task.data,
                begin,
                middle,
                task.function,
                task.grain
            };

            ParallelForTask<T> right{
                task.data,
                middle,
                end,
                task.function,
                task.grain
            };

            Job* left_job =
                CreateJobAsChild(
                    job,
                    &ParallelForTaskFunction<T>,
                    left
                );

            Job* right_job =
                CreateJobAsChild(
                    job,
                    &ParallelForTaskFunction<T>,
                    right
                );

            Run(left_job);
            Run(right_job);
        }

    } // namespace detail


    template <typename T>
    Job* CreateJob(
        Job::JobFunction function,
        const T& payload
    ) {
        static_assert(
            std::is_trivially_copyable_v<T>,
            "Job payload must be trivially copyable."
        );

        static_assert(
            sizeof(T) <= Job::PAYLOAD_SIZE,
            "Job payload does not fit inside Job::data."
        );

        Job* job = CreateJob(function);

        std::memcpy(
            job->data,
            &payload,
            sizeof(T)
        );

        return job;
    }


    template <typename T>
    Job* CreateJobAsChild(
        Job* parent,
        Job::JobFunction function,
        const T& payload
    ) {
        static_assert(
            std::is_trivially_copyable_v<T>,
            "Job payload must be trivially copyable."
        );

        static_assert(
            sizeof(T) <= Job::PAYLOAD_SIZE,
            "Job payload does not fit inside Job::data."
        );

        Job* job =
            CreateJobAsChild(parent, function);

        std::memcpy(
            job->data,
            &payload,
            sizeof(T)
        );

        return job;
    }


    template <typename T>
    void ParallelFor(
        T* data,
        std::uint32_t count,
        ParallelForFunction<T> function,
        std::uint32_t grain
    ) {
        assert(function != nullptr);

        if (count == 0) {
            return;
        }

        if (grain == 0) {
            grain = 1;
        }

        detail::ParallelForTask<T> root_payload{
            data,
            0,
            count,
            function,
            grain
        };

        Job* root =
            CreateJob(
                &detail::ParallelForTaskFunction<T>,
                root_payload
            );

        Run(root);
        Wait(root);
    }

} // namespace JobSystem
