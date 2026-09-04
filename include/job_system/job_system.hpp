#pragma once

#include "job.hpp"
#include "job_allocator.hpp"
#include "work_stealing_queue.hpp"

#include <vector>
#include <thread>
#include <atomic>
#include <memory>
#include <random>
#include <immintrin.h>
#include <functional>
#include <cassert>

namespace job_system {

// Fast thread-local XorShift PRNG for zero-contention random victim selection
class FastRandom {
public:
    explicit FastRandom(uint64_t seed = 88172645463325252ULL) : m_state(seed ? seed : 1) {}

    uint32_t Next(uint32_t min, uint32_t max) {
        m_state ^= m_state >> 12;
        m_state ^= m_state << 25;
        m_state ^= m_state >> 27;
        uint64_t val = m_state * 0x2545F4914F6CDD1DULL;
        return min + static_cast<uint32_t>(val % (max - min));
    }

private:
    uint64_t m_state;
};

template <size_t QueueCapacity = 4096, MemoryOrderingPolicy Policy = MemoryOrderingPolicy::Weak>
class JobSystem {
public:
    using QueueType = WorkStealingQueue<QueueCapacity, Policy>;

    explicit JobSystem(unsigned int numThreads = 0)
        : m_running(true), m_workerCount(0) {
        
        unsigned int hwThreads = std::thread::hardware_concurrency();
        if (hwThreads == 0) hwThreads = 4;
        
        // If unspecified, use all logical cores: N-1 worker threads + main thread = N workers
        unsigned int totalWorkers = (numThreads == 0) ? hwThreads : numThreads;
        if (totalWorkers < 1) totalWorkers = 1;

        m_totalWorkerCount = totalWorkers;
        m_workerCount = totalWorkers - 1;

        // Allocate per-worker queues
        m_queues.reserve(m_totalWorkerCount);
        for (unsigned int i = 0; i < m_totalWorkerCount; ++i) {
            m_queues.emplace_back(std::make_unique<QueueType>());
        }

        // Register main thread as worker 0
        t_workerIndex = 0;
        t_isRegisteredWorker = true;

        // Spawn remaining N-1 background worker threads
        m_threads.reserve(m_workerCount);
        for (unsigned int i = 1; i < m_totalWorkerCount; ++i) {
            m_threads.emplace_back([this, i]() {
                WorkerThreadLoop(i);
            });
        }
    }

    ~JobSystem() {
        Stop();
    }

    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    void Stop() {
        if (!m_running.load(std::memory_order_relaxed)) {
            return;
        }

        m_running.store(false, std::memory_order_release);

        for (auto& th : m_threads) {
            if (th.joinable()) {
                th.join();
            }
        }
        m_threads.clear();
    }

    unsigned int GetWorkerCount() const {
        return m_totalWorkerCount;
    }

    // --- Job Creation (Part 1 & 2) ---

    Job* CreateJob(JobFunction function) {
        Job* job = AllocateJob();
        job->function = function;
        job->parent = nullptr;
        job->unfinishedJobs.store(1, std::memory_order_relaxed);
        job->continuationCount.store(0, std::memory_order_relaxed);
        return job;
    }

    template <typename T>
    Job* CreateJob(JobFunction function, const T& data) {
        static_assert(sizeof(T) <= Job::MAX_JOB_DATA_SIZE, "Job data exceeds in-place data buffer");
        static_assert(std::is_trivially_copyable_v<T>, "Job data must be trivially copyable for memcpy");

        Job* job = AllocateJob();
        job->function = function;
        job->parent = nullptr;
        job->unfinishedJobs.store(1, std::memory_order_relaxed);
        job->continuationCount.store(0, std::memory_order_relaxed);
        std::memcpy(job->data, &data, sizeof(T));
        return job;
    }

    // Modern C++ lambda support for CreateJob (trivially copyable lambdas <= 48 bytes)
    template <typename Lambda>
    Job* CreateJobLambda(Lambda&& lambda) {
        using Decayed = std::decay_t<Lambda>;
        static_assert(sizeof(Decayed) <= Job::MAX_JOB_DATA_SIZE, "Lambda capture size exceeds in-place Job data buffer");
        static_assert(std::is_trivially_copyable_v<Decayed>, "Lambda must be trivially copyable");

        auto trampoline = [](Job*, const void* data) {
            auto& fn = *reinterpret_cast<const Decayed*>(data);
            fn();
        };

        return CreateJob(trampoline, lambda);
    }

    // --- Dynamic Parallelism: Child Job Creation (Part 1 & 4) ---

    Job* CreateJobAsChild(Job* parent, JobFunction function) {
        parent->unfinishedJobs.fetch_add(1, std::memory_order_acq_rel);

        Job* job = AllocateJob();
        job->function = function;
        job->parent = parent;
        job->unfinishedJobs.store(1, std::memory_order_relaxed);
        job->continuationCount.store(0, std::memory_order_relaxed);
        return job;
    }

    template <typename T>
    Job* CreateJobAsChild(Job* parent, JobFunction function, const T& data) {
        static_assert(sizeof(T) <= Job::MAX_JOB_DATA_SIZE, "Job data exceeds in-place data buffer");
        static_assert(std::is_trivially_copyable_v<T>, "Job data must be trivially copyable for memcpy");

        parent->unfinishedJobs.fetch_add(1, std::memory_order_acq_rel);

        Job* job = AllocateJob();
        job->function = function;
        job->parent = parent;
        job->unfinishedJobs.store(1, std::memory_order_relaxed);
        job->continuationCount.store(0, std::memory_order_relaxed);
        std::memcpy(job->data, &data, sizeof(T));
        return job;
    }

    template <typename Lambda>
    Job* CreateJobAsChildLambda(Job* parent, Lambda&& lambda) {
        using Decayed = std::decay_t<Lambda>;
        static_assert(sizeof(Decayed) <= Job::MAX_JOB_DATA_SIZE, "Lambda capture size exceeds in-place Job data buffer");
        static_assert(std::is_trivially_copyable_v<Decayed>, "Lambda must be trivially copyable");

        auto trampoline = [](Job*, const void* data) {
            auto& fn = *reinterpret_cast<const Decayed*>(data);
            fn();
        };

        return CreateJobAsChild(parent, trampoline, lambda);
    }

    // --- Continuations / Dependencies (Part 5) ---

    void AddContinuation(Job* ancestor, Job* continuation) {
        const int32_t count = ancestor->continuationCount.fetch_add(1, std::memory_order_acq_rel);
        assert(count < static_cast<int32_t>(Job::MAX_CONTINUATIONS) && "Max continuations exceeded for Job");
        ancestor->continuations[count] = continuation;
    }

    // --- Job Submission and Scheduling ---

    void Run(Job* job) {
        QueueType* queue = GetCurrentThreadQueue();
        while (!queue->Push(job)) {
            // Queue full backoff: help execute a job to drain queue
            Job* helpJob = queue->Pop();
            if (helpJob) {
                Execute(helpJob);
            } else {
                _mm_pause();
            }
        }
    }

    void Execute(Job* job) {
        (job->function)(job, job->data);
        Finish(job);
    }

    void Finish(Job* job) {
        const int32_t unfinishedJobs = job->unfinishedJobs.fetch_sub(1, std::memory_order_acq_rel);

        if (unfinishedJobs == 1) { // Transitioned to 0
            // Recursively signal parent completion
            if (job->parent) {
                Finish(job->parent);
            }

            // Fire registered continuations
            const int32_t count = job->continuationCount.load(std::memory_order_acquire);
            for (int32_t i = 0; i < count; ++i) {
                Run(job->continuations[i]);
            }
        }
    }

    // --- Active Waiting / Work-Stealing Loop (Part 1) ---

    bool HasJobCompleted(const Job* job) const {
        return job->unfinishedJobs.load(std::memory_order_acquire) <= 0;
    }

    /**
     * @brief Wait for a job to complete.
     * Crucial optimization: the waiting thread does NOT sleep on a mutex.
     * Instead, it actively executes jobs from its own queue or steals jobs
     * from other workers, completely preventing starvation and deadlocks.
     */
    void Wait(const Job* job) {
        while (!HasJobCompleted(job)) {
            Job* nextJob = GetJob();
            if (nextJob) {
                Execute(nextJob);
            } else {
                _mm_pause();
            }
        }
    }

    Job* GetJob() {
        QueueType* myQueue = GetCurrentThreadQueue();

        // 1. Try local LIFO pop (optimal cache locality)
        Job* job = myQueue->Pop();
        if (job) {
            return job;
        }

        // 2. Local queue empty: try stealing FIFO from a random victim
        if (m_totalWorkerCount > 1) {
            uint32_t myIndex = GetCurrentThreadIndex();
            uint32_t randomIndex = t_rng.Next(0, m_totalWorkerCount);

            if (randomIndex == myIndex) {
                randomIndex = (randomIndex + 1) % m_totalWorkerCount;
            }

            Job* stolenJob = m_queues[randomIndex]->Steal();
            if (stolenJob) {
                return stolenJob;
            }
        }

        return nullptr;
    }

private:
    QueueType* GetCurrentThreadQueue() {
        uint32_t index = GetCurrentThreadIndex();
        return m_queues[index].get();
    }

    uint32_t GetCurrentThreadIndex() {
        if (!t_isRegisteredWorker) {
            // External un-registered thread: map to main thread queue 0
            return 0;
        }
        return t_workerIndex;
    }

    void WorkerThreadLoop(unsigned int threadIndex) {
        t_workerIndex = threadIndex;
        t_isRegisteredWorker = true;

        while (m_running.load(std::memory_order_relaxed)) {
            Job* job = GetJob();
            if (job) {
                Execute(job);
            } else {
                // Cooperative idling backoff
                _mm_pause();
            }
        }
    }

    std::atomic<bool> m_running{false};
    unsigned int m_totalWorkerCount{0};
    unsigned int m_workerCount{0};
    std::vector<std::unique_ptr<QueueType>> m_queues;
    std::vector<std::thread> m_threads;

    static thread_local uint32_t t_workerIndex;
    static thread_local bool t_isRegisteredWorker;
    static thread_local FastRandom t_rng;
};

template <size_t QueueCapacity, MemoryOrderingPolicy Policy>
thread_local uint32_t JobSystem<QueueCapacity, Policy>::t_workerIndex = 0;

template <size_t QueueCapacity, MemoryOrderingPolicy Policy>
thread_local bool JobSystem<QueueCapacity, Policy>::t_isRegisteredWorker = false;

template <size_t QueueCapacity, MemoryOrderingPolicy Policy>
thread_local FastRandom JobSystem<QueueCapacity, Policy>::t_rng;

using DefaultJobSystem = JobSystem<4096, MemoryOrderingPolicy::Weak>;

} // namespace job_system
