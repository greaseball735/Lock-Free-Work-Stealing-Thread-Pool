#include "job_system.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>

#if defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
#endif


namespace {

    // =========================================================================
    // Global scheduler state
    // =========================================================================

    std::uint32_t g_num_threads = 0;

    std::vector<std::thread> g_worker_threads;

    std::atomic<bool> g_system_active{false};

    std::vector<std::unique_ptr<WorkStealingQueue>> g_queues;


    // =========================================================================
    // Thread-local scheduler state
    // =========================================================================

    thread_local WorkStealingQueue* t_queue = nullptr;

    thread_local std::uint32_t t_thread_id = 0;

    // -------------------------------------------------------------------------
    // Thread-local allocator
    //
    // This is intentionally not a general-purpose allocator. It is simply
    // a per-thread array of Job descriptors, eliminating malloc/free from
    // the scheduler's normal path.
    // -------------------------------------------------------------------------

    thread_local Job
        t_job_allocator[JobSystemConfig::MAX_JOBS_PER_THREAD];

    thread_local std::uint32_t
        t_allocated_jobs = 0;


    // =========================================================================
    // Tiny per-thread random generator
    // =========================================================================
    //
    // std::rand() is global state and introduces unnecessary contention.
    // A tiny xorshift generator is enough for victim selection.
    // =========================================================================

    thread_local std::uint32_t t_rng_state = 0x9E3779B9u;

    std::uint32_t RandomU32() noexcept {
        std::uint32_t x = t_rng_state;

        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;

        t_rng_state = x;

        return x;
    }


    // =========================================================================
    // Optional CPU affinity
    // =========================================================================

    void PinCurrentThread(std::uint32_t thread_id) noexcept {

#if defined(__linux__)

        if constexpr (JobSystemConfig::PIN_WORKERS) {

            const unsigned int cpu_count =
                std::thread::hardware_concurrency();

            if (cpu_count == 0) {
                return;
            }

            const unsigned int cpu =
                thread_id % cpu_count;

            cpu_set_t cpuset;

            CPU_ZERO(&cpuset);
            CPU_SET(cpu, &cpuset);

            (void)pthread_setaffinity_np(
                pthread_self(),
                sizeof(cpuset),
                &cpuset
            );
        }

#else

        (void)thread_id;

#endif
    }


    // =========================================================================
    // Job completion
    // =========================================================================

    void Finish(Job* job) noexcept {

        const std::int32_t previous =
            job->unfinished_jobs.fetch_sub(
                1,
                std::memory_order_acq_rel
            );

        const std::int32_t remaining =
            previous - 1;

        assert(remaining >= 0);

        if (remaining == 0) {

            Job* parent = job->parent;

            if (parent != nullptr) {
                Finish(parent);
            }
        }
    }


    // =========================================================================
    // Work acquisition
    // =========================================================================

    Job* TrySteal(std::uint32_t thread_id) noexcept {

        if (g_num_threads <= 1) {
            return nullptr;
        }

        // Randomized starting point.
        const std::uint32_t start =
            RandomU32() % g_num_threads;

        for (std::uint32_t i = 0; i < g_num_threads; ++i) {

            const std::uint32_t victim =
                (start + i) % g_num_threads;

            if (victim == thread_id) {
                continue;
            }

            if (Job* job = g_queues[victim]->Steal()) {
                return job;
            }
        }

        return nullptr;
    }


    Job* TryGetJob(std::uint32_t thread_id) noexcept {

        // Prefer our own queue.
        if (Job* job = t_queue->Pop()) {
            return job;
        }

        // Then steal.
        return TrySteal(thread_id);
    }


    // =========================================================================
    // Worker loop
    // =========================================================================

    void WorkerThreadLoop(std::uint32_t thread_id) {

        t_thread_id = thread_id;

        // Newly created worker thread => fresh allocator.
        t_allocated_jobs = 0;

        t_queue = g_queues[thread_id].get();

        // Give every worker a different RNG seed.
        t_rng_state ^=
            0x9E3779B9u *
            (thread_id + 1);

        PinCurrentThread(thread_id);

        std::uint32_t idle_iterations = 0;

        while (g_system_active.load(std::memory_order_acquire)) {

            if (Job* job = TryGetJob(thread_id)) {

                idle_iterations = 0;

                JobSystem::Execute(job);

                continue;
            }

            // No work currently available.
            //
            // Spin for a while because work may arrive very soon,
            // then yield so a completely idle system doesn't consume
            // a full core indefinitely.
            ++idle_iterations;

            _mm_pause();

            if (idle_iterations >=
                JobSystemConfig::SPIN_BEFORE_YIELD) {

                std::this_thread::yield();

                idle_iterations = 0;
            }
        }
    }

} // anonymous namespace


// =============================================================================
// WorkStealingQueue
// =============================================================================

void WorkStealingQueue::Push(Job* job) noexcept {

    assert(job != nullptr);

    const std::int64_t b =
        bottom.load(std::memory_order_relaxed);

    const std::int64_t t =
        top.load(std::memory_order_acquire);

    // The queue is bounded. Overwriting an active entry would be catastrophic.
    //
    // Because only the owner modifies bottom, this check is safe for the
    // intended single-owner model.
    if (static_cast<std::uint64_t>(b - t) >=
        JobSystemConfig::QUEUE_CAPACITY) {

        std::cerr
            << "FATAL: WorkStealingQueue capacity exhausted.\n"
            << "Increase QUEUE_CAPACITY or reduce the number of "
               "simultaneously queued jobs.\n";

        std::abort();
    }

    ring_buffer[
        static_cast<std::uint64_t>(b) & MASK
    ] = job;

    // Publish the job before publishing bottom.
    std::atomic_thread_fence(
        std::memory_order_release
    );

    bottom.store(
        b + 1,
        std::memory_order_relaxed
    );
}


Job* WorkStealingQueue::Pop() noexcept {

    std::int64_t b =
        bottom.load(std::memory_order_relaxed) - 1;

    bottom.store(
        b,
        std::memory_order_relaxed
    );

    // Important ordering point in the Chase-Lev algorithm.
    std::atomic_thread_fence(
        std::memory_order_seq_cst
    );

    std::int64_t t =
        top.load(std::memory_order_relaxed);

    if (t <= b) {

        Job* job =
            ring_buffer[
                static_cast<std::uint64_t>(b) & MASK
            ];

        // Last element:
        //
        // Another worker may concurrently steal it. Exactly one side must win.
        if (t == b) {

            if (!top.compare_exchange_strong(
                    t,
                    t + 1,
                    std::memory_order_seq_cst,
                    std::memory_order_relaxed)) {

                job = nullptr;
            }

            // Restore bottom regardless of who won.
            bottom.store(
                b + 1,
                std::memory_order_relaxed
            );
        }

        return job;
    }

    // Queue became empty.
    bottom.store(
        b + 1,
        std::memory_order_relaxed
    );

    return nullptr;
}


Job* WorkStealingQueue::Steal() noexcept {

    const std::int64_t t =
        top.load(std::memory_order_acquire);

    std::atomic_thread_fence(
        std::memory_order_seq_cst
    );

    const std::int64_t b =
        bottom.load(std::memory_order_acquire);

    if (t >= b) {
        return nullptr;
    }

    Job* job =
        ring_buffer[
            static_cast<std::uint64_t>(t) & MASK
        ];

    // Exactly one thief wins a slot.
    if (!top.compare_exchange_strong(
            const_cast<std::int64_t&>(t),
            t + 1,
            std::memory_order_seq_cst,
            std::memory_order_relaxed)) {

        return nullptr;
    }

    return job;
}


std::int64_t WorkStealingQueue::ApproximateSize() const noexcept {

    const std::int64_t t =
        top.load(std::memory_order_relaxed);

    const std::int64_t b =
        bottom.load(std::memory_order_relaxed);

    return std::max<std::int64_t>(
        0,
        b - t
    );
}


// =============================================================================
// JobSystem
// =============================================================================

namespace JobSystem {


void Initialize(std::uint32_t worker_count) {

    bool expected = false;

    if (!g_system_active.compare_exchange_strong(
            expected,
            true,
            std::memory_order_acq_rel)) {

        // Already initialized.
        return;
    }

    if (worker_count == 0) {

        worker_count =
            std::thread::hardware_concurrency();

        if (worker_count == 0) {
            worker_count = 1;
        }
    }

    g_num_threads = worker_count;

    g_queues.clear();
    g_queues.reserve(g_num_threads);

    for (std::uint32_t i = 0;
         i < g_num_threads;
         ++i) {

        g_queues.emplace_back(
            std::make_unique<WorkStealingQueue>()
        );
    }

    // Calling/main thread is worker 0.
    t_thread_id = 0;
    t_queue = g_queues[0].get();

    // Allow re-initialization of the main thread.
    t_allocated_jobs = 0;

    t_rng_state =
        0xA341316Cu;

    // Worker 0 is the calling thread.
    //
    // Remaining workers are ordinary worker threads.
    g_worker_threads.clear();

    g_worker_threads.reserve(
        g_num_threads > 1
            ? g_num_threads - 1
            : 0
    );

    for (std::uint32_t i = 1;
         i < g_num_threads;
         ++i) {

        g_worker_threads.emplace_back(
            WorkerThreadLoop,
            i
        );
    }
}


void Shutdown() {

    if (!g_system_active.exchange(
            false,
            std::memory_order_acq_rel)) {

        return;
    }

    for (std::thread& worker : g_worker_threads) {

        if (worker.joinable()) {
            worker.join();
        }
    }

    g_worker_threads.clear();

    g_queues.clear();

    g_num_threads = 0;

    t_queue = nullptr;
    t_thread_id = 0;
    t_allocated_jobs = 0;
}


bool IsInitialized() noexcept {
    return g_system_active.load(
        std::memory_order_acquire
    );
}


std::uint32_t ThreadCount() noexcept {
    return g_num_threads;
}


// =============================================================================
// Allocation
// =============================================================================

Job* AllocateJob() {

    const std::uint32_t index =
        t_allocated_jobs++;

    if (index >=
        JobSystemConfig::MAX_JOBS_PER_THREAD) {

        std::cerr
            << "FATAL: Thread-local job allocator exhausted.\n"
            << "Increase MAX_JOBS_PER_THREAD or reduce the number of "
               "simultaneously allocated jobs.\n";

        std::abort();
    }

    return &t_job_allocator[index];
}


// =============================================================================
// Job creation
// =============================================================================

Job* CreateJob(Job::JobFunction function) {

    assert(IsInitialized());
    assert(function != nullptr);

    Job* job = AllocateJob();

    job->function = function;
    job->parent = nullptr;

    job->unfinished_jobs.store(
        1,
        std::memory_order_relaxed
    );

    return job;
}


Job* CreateJobAsChild(
    Job* parent,
    Job::JobFunction function
) {

    assert(IsInitialized());
    assert(parent != nullptr);
    assert(function != nullptr);

    // The parent is still executing while children are created, so it
    // cannot reach zero until this function returns. Allocate first,
    // then increment the parent's unfinished count.
    Job* job = AllocateJob();

    parent->unfinished_jobs.fetch_add(
        1,
        std::memory_order_relaxed
    );

    job->function = function;
    job->parent = parent;

    job->unfinished_jobs.store(
        1,
        std::memory_order_relaxed
    );

    return job;
}


// =============================================================================
// Scheduling
// =============================================================================

void Run(Job* job) {

    assert(IsInitialized());
    assert(job != nullptr);
    assert(t_queue != nullptr);

    t_queue->Push(job);
}


void Execute(Job* job) {

    assert(job != nullptr);
    assert(job->function != nullptr);

    job->function(
        job,
        job->data
    );

    Finish(job);
}


// =============================================================================
// Completion
// =============================================================================

bool HasJobCompleted(const Job* job) {

    assert(job != nullptr);

    return job->unfinished_jobs.load(
        std::memory_order_acquire
    ) == 0;
}


void Wait(const Job* job) {

    assert(IsInitialized());
    assert(job != nullptr);
    assert(t_queue != nullptr);

    std::uint32_t idle_iterations = 0;

    while (!HasJobCompleted(job)) {

        if (Job* next_job =
                TryGetJob(t_thread_id)) {

            idle_iterations = 0;

            Execute(next_job);

            continue;
        }

        ++idle_iterations;

        _mm_pause();

        if (idle_iterations >=
            JobSystemConfig::SPIN_BEFORE_YIELD) {

            std::this_thread::yield();

            idle_iterations = 0;
        }
    }
}


} // namespace JobSystem
