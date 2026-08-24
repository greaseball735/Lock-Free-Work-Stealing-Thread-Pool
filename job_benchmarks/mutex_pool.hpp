#pragma once
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <stdexcept>

class MutexThreadPool {
public:
    explicit MutexThreadPool(std::size_t n) : stopping_(false), active_(0) {
        if (n == 0) n = 1;
        workers_.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            workers_.emplace_back([this] { worker_loop(); });
    }

    ~MutexThreadPool() { shutdown(); }

    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stopping_) throw std::runtime_error("submit after shutdown");
            ++active_;
            q_.push(std::move(f));
        }
        cv_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [this] { return q_.empty() && active_ == 0; });
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
        workers_.clear();
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [this] { return stopping_ || !q_.empty(); });
                if (stopping_ && q_.empty()) return;
                f = std::move(q_.front());
                q_.pop();
            }

            f();

            {
                std::lock_guard<std::mutex> lk(m_);
                --active_;
                if (q_.empty() && active_ == 0) done_.notify_all();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> q_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    bool stopping_;
    std::size_t active_;
};
