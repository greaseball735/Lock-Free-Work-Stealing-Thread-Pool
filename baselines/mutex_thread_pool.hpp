#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

namespace baselines {

/**
 * @brief Standard generic mutex-based thread pool.
 *
 * Characteristics of generic pools:
 * - Single shared FIFO task queue (`std::queue<std::function<void()>>`).
 * - All threads contend on a single `std::mutex` for both push and pop.
 * - Dynamic heap allocations inside `std::function` and queue node allocations.
 * - Worker threads block on `std::condition_variable` when waiting for work.
 */
class MutexThreadPool {
public:
    explicit MutexThreadPool(unsigned int threads = std::thread::hardware_concurrency())
        : m_stop(false), m_inFlightTasks(0) {
        if (threads == 0) threads = 4;
        m_workerCount = threads;

        for (unsigned int i = 0; i < threads; ++i) {
            m_workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(m_queueMutex);
                        m_cv.wait(lock, [this]() {
                            return m_stop.load(std::memory_order_relaxed) || !m_tasks.empty();
                        });

                        if (m_stop.load(std::memory_order_relaxed) && m_tasks.empty()) {
                            return;
                        }

                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }

                    task();

                    if (m_inFlightTasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        std::lock_guard<std::mutex> lock(m_waitMutex);
                        m_waitCv.notify_all();
                    }
                }
            });
        }
    }

    ~MutexThreadPool() {
        m_stop.store(true, std::memory_order_release);
        m_cv.notify_all();
        for (std::thread& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template <class F>
    void Enqueue(F&& f) {
        m_inFlightTasks.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(m_queueMutex);
            m_tasks.emplace(std::forward<F>(f));
        }
        m_cv.notify_one();
    }

    void WaitAll() {
        std::unique_lock<std::mutex> lock(m_waitMutex);
        m_waitCv.wait(lock, [this]() {
            return m_inFlightTasks.load(std::memory_order_acquire) == 0;
        });
    }

    unsigned int GetWorkerCount() const {
        return m_workerCount;
    }

private:
    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;

    std::mutex m_queueMutex;
    std::condition_variable m_cv;

    std::mutex m_waitMutex;
    std::condition_variable m_waitCv;

    std::atomic<bool> m_stop;
    std::atomic<size_t> m_inFlightTasks;
    unsigned int m_workerCount;
};

} 
