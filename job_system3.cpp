#include "job_system.hpp"
#include <immintrin.h> // For _mm_pause
#include <pthread.h>
#include <sched.h>
#include <iostream>
namespace {
    uint32_t g_num_threads = 0;
    std::vector<std::thread> g_worker_threads;
    std::atomic<bool> g_system_active{false};

    // Thread-local data
    thread_local WorkStealingQueue* t_queue = nullptr;
    
    // Lock-free ring-buffer allocator per thread to eliminate heap allocation latency
    static constexpr uint32_t MAX_JOBS_PER_THREAD = 4096;
    thread_local Job t_job_allocator[MAX_JOBS_PER_THREAD];
    thread_local uint32_t t_allocated_jobs = 0;

    std::vector<WorkStealingQueue*> g_queues;

    void WorkerThreadLoop(uint32_t thread_id) {
        // Pin thread to logical core strictly (SMP symmetric mapping, no NUMA topology logic)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        t_queue = g_queues[thread_id];

        while (g_system_active.load(std::memory_order_relaxed)) {
            Job* job = t_queue->Pop();
            if (!job) {
                // Steal randomly from other queues if local queue is empty
                uint32_t random_index = rand() % g_num_threads;
                if (random_index != thread_id) {
                    job = g_queues[random_index]->Steal();
                }
            }

            if (job) {
                JobSystem::Execute(job);
            } else {
                _mm_pause(); // Spin-backoff to prevent pipeline starvation
            }
        }
    }
    
    void Finish(Job* job) {
        // Decrement sequence with release-acquire semantics mapping to the Happens-Before model
        const int32_t unfinished = job->unfinished_jobs.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (unfinished == 0) {
            if (job->parent) {
                Finish(job->parent);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Lock-Free Deque Implementation
// -----------------------------------------------------------------------------
void WorkStealingQueue::Push(Job* job) {
    int32_t b = bottom.load(std::memory_order_relaxed);
    ring_buffer[b & MASK] = job;
    
    std::atomic_thread_fence(std::memory_order_release);
    bottom.store(b + 1, std::memory_order_relaxed);
}

Job* WorkStealingQueue::Pop() {
    int32_t b = bottom.load(std::memory_order_relaxed) - 1;
    bottom.store(b, std::memory_order_relaxed);
    
    std::atomic_thread_fence(std::memory_order_seq_cst); // Enforce global ordering against Steal
    int32_t t = top.load(std::memory_order_relaxed);

    if (t <= b) {
        Job* job = ring_buffer[b & MASK];
        if (t == b) {
            if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                job = nullptr; // Conflict with concurrent Steal
            }
            bottom.store(b + 1, std::memory_order_relaxed);
        }
        return job;
    } else {
        bottom.store(b + 1, std::memory_order_relaxed);
        return nullptr;
    }
}

Job* WorkStealingQueue::Steal() {
    int32_t t = top.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst); // Enforce global ordering against Pop
    int32_t b = bottom.load(std::memory_order_acquire);

    if (t < b) {
        Job* job = ring_buffer[t & MASK];
        if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
            return nullptr;
        }
        return job;
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// JobSystem Core API Implementation
// -----------------------------------------------------------------------------
void JobSystem::Initialize() {
    g_num_threads = std::thread::hardware_concurrency();
    g_system_active.store(true, std::memory_order_relaxed);

    g_queues.resize(g_num_threads);
    for (uint32_t i = 0; i < g_num_threads; ++i) {
        g_queues[i] = new WorkStealingQueue();
    }

    t_queue = g_queues[0]; // Main thread adopts queue 0
    
    // Spawn worker threads
    for (uint32_t i = 1; i < g_num_threads; ++i) {
        g_worker_threads.emplace_back(WorkerThreadLoop, i);
    }
}

void JobSystem::Shutdown() {
    g_system_active.store(false, std::memory_order_release);
    for (auto& thread : g_worker_threads) {
        thread.join();
    }
    for (auto queue : g_queues) {
        delete queue;
    }
}

Job* AllocateJob() {
    uint32_t index = t_allocated_jobs++;
    if (index >= MAX_JOBS_PER_THREAD) {
        // Prevent silent memory corruption wrap-around
        std::cerr << "CRITICAL: Thread-local job allocator exhausted!\n";
        std::exit(1);
    }
    return &t_job_allocator[index];
}
// Job* JobSystem::AllocateJob() {
//     uint32_t index = t_allocated_jobs++;
//     return &t_job_allocator[index & (MAX_JOBS_PER_THREAD - 1)];
// }

Job* JobSystem::CreateJob(Job::JobFunction function) {
    Job* job = AllocateJob();
    job->function = function;
    job->parent = nullptr;
    job->unfinished_jobs.store(1, std::memory_order_relaxed);
    return job;
}

Job* JobSystem::CreateJobAsChild(Job* parent, Job::JobFunction function) {
    parent->unfinished_jobs.fetch_add(1, std::memory_order_relaxed);
    Job* job = AllocateJob();
    job->function = function;
    job->parent = parent;
    job->unfinished_jobs.store(1, std::memory_order_relaxed);
    return job;
}

void JobSystem::Run(Job* job) {
    t_queue->Push(job);
}

void JobSystem::Execute(Job* job) {
    (job->function)(job, job->data);
    Finish(job);
}

bool JobSystem::HasJobCompleted(const Job* job) {
    return job->unfinished_jobs.load(std::memory_order_acquire) <= 0;
}

void JobSystem::Wait(const Job* job) {
    while (!HasJobCompleted(job)) {
        Job* nextJob = t_queue->Pop();
        if (!nextJob) {
            uint32_t random_index = rand() % g_num_threads;
            nextJob = g_queues[random_index]->Steal();
        }
        if (nextJob) {
            Execute(nextJob);
        } else {
            _mm_pause();
        }
    }
}
