#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

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
            worker.join();
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
        // Active wait for simplicity in benchmarking against job system Wait()
        while (pending_tasks.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield(); 
        }
    }
};
