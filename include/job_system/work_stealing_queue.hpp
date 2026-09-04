#pragma once

#include "job.hpp"
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <new>

namespace job_system {

/**
 * @brief Configurable memory model semantics for work-stealing queue operations.
 *
 * Allows experimenting with the performance impact of standard weak memory model
 * guarantees (C++11 acquire/release + seq_cst fences) vs x86 TSO-specialized barriers.
 */
enum class MemoryOrderingPolicy {
    // Standard-conforming weak memory model (safe on ARM, POWER, RISC-V, and x86)
    Weak,

    // x86 TSO optimized: relies on x86 hardware store-store and load-load ordering,
    // using compiler barriers for those, but retaining the essential Store-Load fence on Pop()
    TSO_Optimized,

    // Experimental relaxed mode: eliminates the Store-Load fence on Pop() to measure
    // pure barrier overhead (WARNING: may exhibit races under high contention)
    Relaxed_Experiment
};

/**
 * @brief Lock-free circular work-stealing deque based on Chase & Lev (2005)
 *        as detailed in Part 3 of the blog series.
 *
 * Properties:
 * - Single-Producer, Multi-Consumer (SPMC) deque.
 * - Private end (bottom): Push() and Pop() called exclusively by queue owner thread (LIFO order).
 * - Public end (top): Steal() called concurrently by thief threads (FIFO order).
 * - Top and bottom are placed on distinct 64-byte cache lines to eliminate false sharing.
 */
template <size_t Capacity = 4096, MemoryOrderingPolicy Policy = MemoryOrderingPolicy::Weak>
class WorkStealingQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

public:
    static constexpr size_t MASK = Capacity - 1;

    WorkStealingQueue() {
        m_top.store(0, std::memory_order_relaxed);
        m_bottom.store(0, std::memory_order_relaxed);
        for (size_t i = 0; i < Capacity; ++i) {
            m_jobs[i] = nullptr;
        }
    }

    ~WorkStealingQueue() = default;

    WorkStealingQueue(const WorkStealingQueue&) = delete;
    WorkStealingQueue& operator=(const WorkStealingQueue&) = delete;

    /**
     * @brief Pushes a job to the private (LIFO) end of the queue.
     * Only called by the worker thread that owns this queue.
     */
    bool Push(Job* job) {
        int64_t b = m_bottom.load(std::memory_order_relaxed);
        int64_t t = m_top.load(std::memory_order_acquire);

        // Check if queue is full
        if (b - t >= static_cast<int64_t>(Capacity)) {
            return false;
        }

        m_jobs[b & MASK] = job;

        if constexpr (Policy == MemoryOrderingPolicy::Weak) {
            // Ensure job payload write is visible before bottom is incremented
            m_bottom.store(b + 1, std::memory_order_release);
        } else if constexpr (Policy == MemoryOrderingPolicy::TSO_Optimized) {
            // On x86 TSO, store-store is hardware-ordered; compiler barrier is sufficient
          std::atomic_signal_fence(std::memory_order_acq_rel);                                                                         
            m_bottom.store(b + 1, std::memory_order_relaxed);
        } else {
            m_bottom.store(b + 1, std::memory_order_relaxed);
        }

        return true;
    }

    /**
     * @brief Pops a job from the private (LIFO) end of the queue.
     * Only called by the worker thread that owns this queue.
     */
    Job* Pop() {
        int64_t b = m_bottom.load(std::memory_order_relaxed) - 1;

        if constexpr (Policy == MemoryOrderingPolicy::Weak) {
            // Weak memory model: store bottom with seq_cst to prevent Store-Load reordering with top
            m_bottom.store(b, std::memory_order_seq_cst);
            int64_t t = m_top.load(std::memory_order_seq_cst);
            return PopInternal(b, t);
        } else if constexpr (Policy == MemoryOrderingPolicy::TSO_Optimized) {
            // x86 TSO: Store followed by Load to different addresses can be reordered by the CPU.
            // On x86, atomic exchange (xchg) acts as an implicit full fence and is faster than mfence.
            m_bottom.exchange(b, std::memory_order_acq_rel);
            int64_t t = m_top.load(std::memory_order_relaxed);
            return PopInternal(b, t);
        } else {
            // Relaxed experiment: no fence between store and load
            m_bottom.store(b, std::memory_order_relaxed);
            int64_t t = m_top.load(std::memory_order_relaxed);
            return PopInternal(b, t);
        }
    }

    /**
     * @brief Steals a job from the public (FIFO) end of the queue.
     * Called concurrently by thief threads attempting work-stealing.
     */
    Job* Steal() {
        int64_t t;
        int64_t b;

        if constexpr (Policy == MemoryOrderingPolicy::Weak) {
            t = m_top.load(std::memory_order_seq_cst);
            b = m_bottom.load(std::memory_order_seq_cst);
        } else if constexpr (Policy == MemoryOrderingPolicy::TSO_Optimized) {
            t = m_top.load(std::memory_order_relaxed);
                std::atomic_signal_fence(std::memory_order_acq_rel);  
            b = m_bottom.load(std::memory_order_relaxed);
        } else {
            t = m_top.load(std::memory_order_relaxed);
            b = m_bottom.load(std::memory_order_relaxed);
        }

        if (t < b) {
            // Non-empty queue: fetch job pointer before attempting CAS
            Job* job = m_jobs[t & MASK];

            // Compete against concurrent thieves and the owner
            if (!m_top.compare_exchange_strong(t, t + 1,
                                                std::memory_order_seq_cst,
                                                std::memory_order_relaxed)) {
                // Lost race to another thief or to owner's Pop()
                return nullptr;
            }

            return job;
        }

        return nullptr;
    }

    bool IsEmpty() const {
        int64_t b = m_bottom.load(std::memory_order_relaxed);
        int64_t t = m_top.load(std::memory_order_relaxed);
        return b <= t;
    }

    size_t Size() const {
        int64_t b = m_bottom.load(std::memory_order_relaxed);
        int64_t t = m_top.load(std::memory_order_relaxed);
        return (b > t) ? static_cast<size_t>(b - t) : 0;
    }

private:
    Job* PopInternal(int64_t b, int64_t t) {
        if (t <= b) {
            // Deque has at least 1 job
            Job* job = m_jobs[b & MASK];

            if (t != b) {
                // More than 1 item remains: no conflict with thieves
                return job;
            }

            // Exactly 1 item remains: owner must arbitrate against potential thieves
            if (!m_top.compare_exchange_strong(t, t + 1,
                                                std::memory_order_seq_cst,
                                                std::memory_order_relaxed)) {
                // Lost race against Steal()
                job = nullptr;
            }

            // Reset bottom to canonical empty state (t + 1)
            m_bottom.store(t + 1, std::memory_order_relaxed);
            return job;
        } else {
            // Deque was already empty
            m_bottom.store(t, std::memory_order_relaxed);
            return nullptr;
        }
    }

    // Cache-line separated members to avoid false sharing
    alignas(64) std::atomic<int64_t> m_top{0};
    alignas(64) std::atomic<int64_t> m_bottom{0};
    alignas(64) Job* m_jobs[Capacity];
};

using DefaultWorkStealingQueue = WorkStealingQueue<4096, MemoryOrderingPolicy::Weak>;

} // namespace job_system
