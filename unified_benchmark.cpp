#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <omp.h>
#include "asymmetric_workloads.hpp"
// =============================================================================
// 1. CUSTOM LOCK-FREE WORK-STEALING JOB SYSTEM
// =============================================================================

struct alignas(64) Job {
    using JobFunction = void(*)(Job*, const void*);
    JobFunction function;
    Job* parent;
    std::atomic<int32_t> unfinished_jobs;

    static constexpr size_t PAYLOAD_SIZE = 64 - sizeof(JobFunction) - sizeof(Job*) - sizeof(std::atomic<int32_t>);
    uint8_t data[PAYLOAD_SIZE];
};

class WorkStealingQueue {
private:
    static constexpr int32_t CAPACITY = 8192;
    static constexpr int32_t MASK = CAPACITY - 1;

    alignas(64) std::atomic<int32_t> top{0};
    alignas(64) std::atomic<int32_t> bottom{0};
    Job* ring_buffer[CAPACITY];

public:
    void Push(Job* job) {
        int32_t b = bottom.load(std::memory_order_relaxed);
        ring_buffer[b & MASK] = job;
        std::atomic_thread_fence(std::memory_order_release);
        bottom.store(b + 1, std::memory_order_relaxed);
    }

    Job* Pop() {
        int32_t b = bottom.load(std::memory_order_relaxed) - 1;
        bottom.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int32_t t = top.load(std::memory_order_relaxed);

        if (t <= b) {
            Job* job = ring_buffer[b & MASK];
            if (t == b) {
                if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                    job = nullptr;
                }
                bottom.store(b + 1, std::memory_order_relaxed);
            }
            return job;
        } else {
            bottom.store(b + 1, std::memory_order_relaxed);
            return nullptr;
        }
    }

    Job* Steal() {
        int32_t t = top.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
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
};

namespace JobSystem {
    static uint32_t g_num_threads = 0;
    static std::vector<std::thread> g_worker_threads;
    static std::atomic<bool> g_system_active{false};

    static thread_local WorkStealingQueue* t_queue = nullptr;
    static constexpr uint32_t MAX_JOBS_PER_THREAD = 8192;
    static thread_local Job t_job_allocator[MAX_JOBS_PER_THREAD];
    static thread_local uint32_t t_allocated_jobs = 0;

    static std::vector<WorkStealingQueue*> g_queues;

    void Execute(Job* job);

    static void Finish(Job* job) {
        const int32_t unfinished = job->unfinished_jobs.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (unfinished == 0 && job->parent) {
            Finish(job->parent);
        }
    }

    static void WorkerThreadLoop(uint32_t thread_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        t_queue = g_queues[thread_id];

        while (g_system_active.load(std::memory_order_relaxed)) {
            Job* job = t_queue->Pop();
            if (!job) {
                uint32_t random_index = rand() % g_num_threads;
                if (random_index != thread_id) {
                    job = g_queues[random_index]->Steal();
                }
            }

            if (job) {
                Execute(job);
            } else {
                _mm_pause();
            }
        }
    }

    void Initialize() {
        g_num_threads = std::thread::hardware_concurrency();
        g_system_active.store(true, std::memory_order_relaxed);

        g_queues.resize(g_num_threads);
        for (uint32_t i = 0; i < g_num_threads; ++i) {
            g_queues[i] = new WorkStealingQueue();
        }

        t_queue = g_queues[0];
        for (uint32_t i = 1; i < g_num_threads; ++i) {
            g_worker_threads.emplace_back(WorkerThreadLoop, i);
        }
    }

    void Shutdown() {
        g_system_active.store(false, std::memory_order_release);
        for (auto& thread : g_worker_threads) {
            if (thread.joinable()) thread.join();
        }
        g_worker_threads.clear();
        for (auto queue : g_queues) {
            delete queue;
        }
        g_queues.clear();
    }

    Job* AllocateJob() {
        uint32_t index = t_allocated_jobs++;
        return &t_job_allocator[index & (MAX_JOBS_PER_THREAD - 1)];
    }

    Job* CreateJob(Job::JobFunction function) {
        Job* job = AllocateJob();
        job->function = function;
        job->parent = nullptr;
        job->unfinished_jobs.store(1, std::memory_order_relaxed);
        return job;
    }

    Job* CreateJobAsChild(Job* parent, Job::JobFunction function) {
        parent->unfinished_jobs.fetch_add(1, std::memory_order_relaxed);
        Job* job = AllocateJob();
        job->function = function;
        job->parent = parent;
        job->unfinished_jobs.store(1, std::memory_order_relaxed);
        return job;
    }

    void Run(Job* job) {
        t_queue->Push(job);
    }

    void Execute(Job* job) {
        (job->function)(job, job->data);
        Finish(job);
    }

    bool HasJobCompleted(const Job* job) {
        return job->unfinished_jobs.load(std::memory_order_acquire) <= 0;
    }

    void Wait(const Job* job) {
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
}

// =============================================================================
// 2. NAIVE MUTEX THREAD POOL (BASELINE 1)
// =============================================================================

class NaiveThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    std::atomic<uint32_t> pending_tasks{0};

public:
    NaiveThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { 
                            return this->stop.load(std::memory_order_relaxed) || !this->tasks.empty(); 
                        });
                        if (this->stop.load(std::memory_order_relaxed) && this->tasks.empty()) {
                            return;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                    pending_tasks.fetch_sub(1, std::memory_order_release);
                }
            });
        }
    }

    ~NaiveThreadPool() {
        stop.store(true, std::memory_order_relaxed);
        condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    void Enqueue(std::function<void()> task) {
        pending_tasks.fetch_add(1, std::memory_order_acquire);
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.push(std::move(task));
        }
        condition.notify_one();
    }

    void WaitAll() {
        while (pending_tasks.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }
};

// =============================================================================
// 3. WORKLOAD KERNELS
// =============================================================================

// Sequential Fibonacci
uint64_t FibSequential(uint64_t n) {
    if (n <= 1) return n;
    return FibSequential(n - 1) + FibSequential(n - 2);
}

// OpenMP Fibonacci
uint64_t FibOMP(uint64_t n, int cutoff) {
    if (n <= 1) return n;
    if (n <= cutoff) return FibSequential(n);
    
    uint64_t x, y;
    #pragma omp task shared(x) firstprivate(n, cutoff)
    x = FibOMP(n - 1, cutoff);
    
    #pragma omp task shared(y) firstprivate(n, cutoff)
    y = FibOMP(n - 2, cutoff);
    
    #pragma omp taskwait
    return x + y;
}

// Work-Stealing Fibonacci Payload & Execution Function
struct FibPayload {
    uint64_t n;
    uint64_t result;
    int cutoff;
};
void FibJobFunc(Job* job, const void* raw_data) {
    FibPayload* payload = const_cast<FibPayload*>(reinterpret_cast<const FibPayload*>(raw_data));
    if (payload->n <= payload->cutoff) {
        payload->result = FibSequential(payload->n);
        return;
    }

    Job* child1 = JobSystem::CreateJobAsChild(job, FibJobFunc);
    FibPayload* p1 = reinterpret_cast<FibPayload*>(child1->data);
    p1->n = payload->n - 1;
    p1->cutoff = payload->cutoff;

    Job* child2 = JobSystem::CreateJobAsChild(job, FibJobFunc);
    FibPayload* p2 = reinterpret_cast<FibPayload*>(child2->data);
    p2->n = payload->n - 2;
    p2->cutoff = payload->cutoff;

    JobSystem::Run(child1);
    JobSystem::Run(child2);

    // FIX: Wait explicitly on the child jobs to reach 0, rather than the parent
    JobSystem::Wait(child1);
    JobSystem::Wait(child2);
    
    payload->result = p1->result + p2->result;
}
// void FibJobFunc(Job* job, const void* raw_data) {
//     FibPayload* payload = const_cast<FibPayload*>(reinterpret_cast<const FibPayload*>(raw_data));
//     if (payload->n <= payload->cutoff) {
//         payload->result = FibSequential(payload->n);
//         return;
//     }

//     Job* child1 = JobSystem::CreateJobAsChild(job, FibJobFunc);
//     FibPayload* p1 = reinterpret_cast<FibPayload*>(child1->data);
//     p1->n = payload->n - 1;
//     p1->cutoff = payload->cutoff;

//     Job* child2 = JobSystem::CreateJobAsChild(job, FibJobFunc);
//     FibPayload* p2 = reinterpret_cast<FibPayload*>(child2->data);
//     p2->n = payload->n - 2;
//     p2->cutoff = payload->cutoff;

//     JobSystem::Run(child1);
//     JobSystem::Run(child2);

//     JobSystem::Wait(job);
//     payload->result = p1->result + p2->result;
// }

// Matrix Multiplication Helper Function
void MatrixMultiply(const float* A, const float* B, float* C, int N, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

struct MatrixPayload {
    const float* A;
    const float* B;
    float* C;
    int N;
    int start_row;
    int end_row;
};

void MatrixJobFunc(Job*, const void* raw_data) {
    const MatrixPayload* p = reinterpret_cast<const MatrixPayload*>(raw_data);
    MatrixMultiply(p->A, p->B, p->C, p->N, p->start_row, p->end_row);
}
// --- Quicksort Job Data ---
struct QsPayload {
    int* arr;
    int low;
    int high;
    int cutoff;
};

void QsJobFunc(Job* job, const void* raw_data) {
    const QsPayload* payload = reinterpret_cast<const QsPayload*>(raw_data);
    int low = payload->low;
    int high = payload->high;
    int cutoff = payload->cutoff;
    int* arr = payload->arr;

    if (high - low < cutoff) {
        QsSequential(arr, low, high, cutoff);
        return;
    }

    int pi = Partition(arr, low, high);

    Job* child1 = JobSystem::CreateJobAsChild(job, QsJobFunc);
    QsPayload* p1 = reinterpret_cast<QsPayload*>(child1->data);
    p1->arr = arr; p1->low = low; p1->high = pi - 1; p1->cutoff = cutoff;

    Job* child2 = JobSystem::CreateJobAsChild(job, QsJobFunc);
    QsPayload* p2 = reinterpret_cast<QsPayload*>(child2->data);
    p2->arr = arr; p2->low = pi + 1; p2->high = high; p2->cutoff = cutoff;

    JobSystem::Run(child1);
    JobSystem::Run(child2);

    JobSystem::Wait(child1);
    JobSystem::Wait(child2);
}

// --- Mandelbrot Job Data ---
struct MandelbrotPayload {
    int* output;
    int width;
    int height;
    int start_x;
    int end_x;
    int start_y;
    int end_y;
    int max_iter;
};

void MandelbrotJobFunc(Job* job, const void* raw_data) {
    const MandelbrotPayload* p = reinterpret_cast<const MandelbrotPayload*>(raw_data);
    ComputeMandelbrotChunk(p->output, p->width, p->height, 
                           p->start_x, p->end_x, p->start_y, p->end_y, p->max_iter);
}
// =============================================================================
// 4. MAIN BENCHMARK HARNESS
// =============================================================================

int main() {
    const uint32_t num_threads = std::thread::hardware_concurrency();
    std::cout << "=========================================================\n";
    std::cout << "  UNIFIED BENCHMARK HARNESS (" << num_threads << " LOGICAL CORES DETECTED)\n";
    std::cout << "=========================================================\n\n";

    // -------------------------------------------------------------------------
    // WORKLOAD 1: MATRIX MULTIPLICATION (1024 x 1024)
    // -------------------------------------------------------------------------
    std::cout << "--- WORKLOAD 1: Dense Matrix Multiplication (1024 x 1024) ---\n";
    constexpr int N = 1024;
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);
    const int chunk = N / num_threads;

    // Baseline 0: Sequential
    auto start = std::chrono::high_resolution_clock::now();
    MatrixMultiply(A.data(), B.data(), C.data(), N, 0, N);
    auto end = std::chrono::high_resolution_clock::now();
    double t_seq_mat = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Baseline 0 (Sequential):            " << t_seq_mat << " ms\n";

    // Baseline 1: Mutex Thread Pool
    std::fill(C.begin(), C.end(), 0.0f);
    {
        NaiveThreadPool pool(num_threads);
        start = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < num_threads; ++i) {
            int r_start = i * chunk;
            int r_end = (i == num_threads - 1) ? N : r_start + chunk;
            pool.Enqueue([&A, &B, &C, N, r_start, r_end]() {
                MatrixMultiply(A.data(), B.data(), C.data(), N, r_start, r_end);
            });
        }
        pool.WaitAll();
        end = std::chrono::high_resolution_clock::now();
    }
    double t_pool_mat = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Baseline 1 (Mutex Thread Pool):     " << t_pool_mat << " ms\n";

    // Baseline 2: OpenMP
    std::fill(C.begin(), C.end(), 0.0f);
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double t_omp_mat = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Baseline 2 (OpenMP Loop Parallel):  " << t_omp_mat << " ms\n";

    // Custom: Lock-Free Work-Stealing System
    std::fill(C.begin(), C.end(), 0.0f);
    JobSystem::Initialize();
    start = std::chrono::high_resolution_clock::now();
    Job* root_mat = JobSystem::CreateJob([](Job*, const void*){});
    for (uint32_t i = 0; i < num_threads; ++i) {
        Job* child = JobSystem::CreateJobAsChild(root_mat, MatrixJobFunc);
        MatrixPayload* p = reinterpret_cast<MatrixPayload*>(child->data);
        p->A = A.data(); p->B = B.data(); p->C = C.data(); p->N = N;
        p->start_row = i * chunk;
        p->end_row = (i == num_threads - 1) ? N : p->start_row + chunk;
        JobSystem::Run(child);
    }
    JobSystem::Run(root_mat);
    JobSystem::Wait(root_mat);
    end = std::chrono::high_resolution_clock::now();
    double t_ws_mat = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Custom (Lock-Free Work-Stealing):   " << t_ws_mat << " ms\n\n";
    JobSystem::Shutdown();

    // -------------------------------------------------------------------------
    // WORKLOAD 2: DYNAMIC RECURSIVE FIBONACCI (N = 42, Cutoff = 22)
    // -------------------------------------------------------------------------
    std::cout << "--- WORKLOAD 2: Dynamic Recursive Fibonacci (N = 42, Cutoff = 22) ---\n";
    constexpr uint64_t FIB_N = 42;
    constexpr int CUTOFF = 22;

    // Baseline 0: Sequential
    start = std::chrono::high_resolution_clock::now();
    uint64_t fib_seq_res = FibSequential(FIB_N);
    end = std::chrono::high_resolution_clock::now();
    double t_seq_fib = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Baseline 0 (Sequential):            " << t_seq_fib << " ms | Result: " << fib_seq_res << "\n";

    // Baseline 1: Mutex Thread Pool (Decomposed at top-level to prevent recursive worker deadlock)
    start = std::chrono::high_resolution_clock::now();
    std::atomic<uint64_t> pool_fib_accum{0};
    {
        NaiveThreadPool pool(num_threads);
        std::function<void(uint64_t, int)> decompose = [&](uint64_t n, int depth) {
            if (depth <= 0 || n <= CUTOFF) {
                pool_fib_accum.fetch_add(FibSequential(n), std::memory_order_relaxed);
                return;
            }
            pool.Enqueue([&decompose, n, depth]() { decompose(n - 1, depth - 1); });
            pool.Enqueue([&decompose, n, depth]() { decompose(n - 2, depth - 1); });
        };
        decompose(FIB_N, 8); // Decompose top 8 levels into pool
        pool.WaitAll();
    }
    end = std::chrono::high_resolution_clock::now();
    double t_pool_fib = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Baseline 1 (Mutex Thread Pool):     " << t_pool_fib << " ms | Result: " << pool_fib_accum.load() << "\n";

    // Baseline 2: OpenMP Tasks
    start = std::chrono::high_resolution_clock::now();
    uint64_t fib_omp_res = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            fib_omp_res = FibOMP(FIB_N, CUTOFF);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    double t_omp_fib = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Baseline 2 (OpenMP Dynamic Tasks):  " << t_omp_fib << " ms | Result: " << fib_omp_res << "\n";

    // Custom: Lock-Free Work-Stealing System
    JobSystem::Initialize();
    start = std::chrono::high_resolution_clock::now();
    Job* root_fib = JobSystem::CreateJob(FibJobFunc);
    FibPayload* p_root = reinterpret_cast<FibPayload*>(root_fib->data);
    p_root->n = FIB_N;
    p_root->cutoff = CUTOFF;
    p_root->result = 0;

    JobSystem::Run(root_fib);
    JobSystem::Wait(root_fib);
    end = std::chrono::high_resolution_clock::now();
    double t_ws_fib = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Custom (Lock-Free Work-Stealing):   " << t_ws_fib << " ms | Result: " << p_root->result << "\n\n";
    JobSystem::Shutdown();
    // -------------------------------------------------------------------------
    // WORKLOAD 3: ASYMMETRIC QUICKSORT (10M Elements, 95% Duplicates)
    // -------------------------------------------------------------------------
    // std::cout << "--- WORKLOAD 3: Asymmetric Quicksort (10M Elements, 95% Duplicates) ---\n";
    // constexpr int QS_SIZE = 100000;
    // std::vector<int> master_arr(QS_SIZE);
    // for (int i = 0; i < QS_SIZE; ++i) {
    //     // Force massive load imbalance with 95% identical elements
    //     master_arr[i] = (rand() % 100 < 5) ? rand() : 42; 
    // }
    // std::vector<int> test_arr = master_arr;

    // // Baseline 0: Sequential
    // start = std::chrono::high_resolution_clock::now();
    // QsSequential(test_arr.data(), 0, QS_SIZE - 1, 128);
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Baseline 0 (Sequential):            " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    // // Baseline 1: Mutex Thread Pool
    // test_arr = master_arr;
    // {
    //     NaiveThreadPool pool(num_threads);
    //     start = std::chrono::high_resolution_clock::now();
    //     // Non-blocking task enqueue prevents deadlock on mutex pool
    //     std::function<void(int, int)> qs_pool = [&](int low, int high) {
    //         if (high - low < 128) { QsSequential(test_arr.data(), low, high, 128); return; }
    //         int pi = Partition(test_arr.data(), low, high);
    //         pool.Enqueue([&qs_pool, low, pi]() { qs_pool(low, pi - 1); });
    //         pool.Enqueue([&qs_pool, pi, high]() { qs_pool(pi + 1, high); });
    //     };
    //     pool.Enqueue([&]() { qs_pool(0, QS_SIZE - 1); });
    //     pool.WaitAll();
    //     end = std::chrono::high_resolution_clock::now();
    // }
    // std::cout << "Baseline 1 (Mutex Thread Pool):     " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    // // Baseline 2: OpenMP Tasks
    // test_arr = master_arr;
    // start = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     QsOMP(test_arr.data(), 0, QS_SIZE - 1, 128);
    // }
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Baseline 2 (OpenMP Dynamic Tasks):  " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    // // Custom: Lock-Free Work-Stealing
    // test_arr = master_arr;
    // JobSystem::Initialize();
    // start = std::chrono::high_resolution_clock::now();
    // Job* root_qs = JobSystem::CreateJob(QsJobFunc);
    // QsPayload* qs_p = reinterpret_cast<QsPayload*>(root_qs->data);
    // qs_p->arr = test_arr.data(); qs_p->low = 0; qs_p->high = QS_SIZE - 1; qs_p->cutoff = 128;
    
    // JobSystem::Run(root_qs);
    // JobSystem::Wait(root_qs);
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Custom (Lock-Free Work-Stealing):   " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n\n";
    // JobSystem::Shutdown();


    // -------------------------------------------------------------------------
    // WORKLOAD 4: MANDELBROT SET (4096 x 4096, Max 10000 Iterations)
    // -------------------------------------------------------------------------
    std::cout << "--- WORKLOAD 4: Mandelbrot Fractal (4096 x 4096) ---\n";
    constexpr int M_SIZE = 2048;
    constexpr int MAX_ITER = 10000;
    constexpr int CHUNK_SIZE = 32; // Highly granular tasks
    std::vector<int> m_output(M_SIZE * M_SIZE, 0);

    // Baseline 0: Sequential
    start = std::chrono::high_resolution_clock::now();
    ComputeMandelbrotChunk(m_output.data(), M_SIZE, M_SIZE, 0, M_SIZE, 0, M_SIZE, MAX_ITER);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Baseline 0 (Sequential):            " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    // Baseline 1: Mutex Thread Pool
    std::fill(m_output.begin(), m_output.end(), 0);
    {
        NaiveThreadPool pool(num_threads);
        start = std::chrono::high_resolution_clock::now();
        for (int y = 0; y < M_SIZE; y += CHUNK_SIZE) {
            for (int x = 0; x < M_SIZE; x += CHUNK_SIZE) {
                int ex = std::min(x + CHUNK_SIZE, M_SIZE);
                int ey = std::min(y + CHUNK_SIZE, M_SIZE);
                pool.Enqueue([&m_output, x, ex, y, ey, M_SIZE, MAX_ITER]() {
                    ComputeMandelbrotChunk(m_output.data(), M_SIZE, M_SIZE, x, ex, y, ey, MAX_ITER);
                });
            }
        }
        pool.WaitAll();
        end = std::chrono::high_resolution_clock::now();
    }
    std::cout << "Baseline 1 (Mutex Thread Pool):     " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    // Baseline 2: OpenMP Dynamic Scheduling
    std::fill(m_output.begin(), m_output.end(), 0);
    start = std::chrono::high_resolution_clock::now();
    // Schedule Dynamic allows OpenMP to act like a work-stealing queue for loop chunks
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE) collapse(2)
    for (int y = 0; y < M_SIZE; ++y) {
        for (int x = 0; x < M_SIZE; ++x) {
            float cr = (x - M_SIZE / 2.0f) * 4.0f / M_SIZE;
            float ci = (y - M_SIZE / 2.0f) * 4.0f / M_SIZE;
            float zr = 0.0f, zi = 0.0f;
            int iter = 0;
            while (zr * zr + zi * zi <= 4.0f && iter < MAX_ITER) {
                float temp = zr * zr - zi * zi + cr;
                zi = 2.0f * zr * zi + ci;
                zr = temp;
                iter++;
            }
            m_output[y * M_SIZE + x] = iter;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Baseline 2 (OpenMP Dynamic Loop):   " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

    // Custom: Lock-Free Work-Stealing
    std::fill(m_output.begin(), m_output.end(), 0);
    JobSystem::Initialize();
    start = std::chrono::high_resolution_clock::now();
    
    Job* root_m = JobSystem::CreateJob([](Job*, const void*){});
    for (int y = 0; y < M_SIZE; y += CHUNK_SIZE) {
        for (int x = 0; x < M_SIZE; x += CHUNK_SIZE) {
            Job* child = JobSystem::CreateJobAsChild(root_m, MandelbrotJobFunc);
            MandelbrotPayload* p = reinterpret_cast<MandelbrotPayload*>(child->data);
            p->output = m_output.data(); p->width = M_SIZE; p->height = M_SIZE;
            p->start_x = x; p->end_x = std::min(x + CHUNK_SIZE, M_SIZE);
            p->start_y = y; p->end_y = std::min(y + CHUNK_SIZE, M_SIZE);
            p->max_iter = MAX_ITER;
            JobSystem::Run(child);
        }
    }
    JobSystem::Run(root_m);
    JobSystem::Wait(root_m);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Custom (Lock-Free Work-Stealing):   " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n\n";
    JobSystem::Shutdown();
    return 0;
}
