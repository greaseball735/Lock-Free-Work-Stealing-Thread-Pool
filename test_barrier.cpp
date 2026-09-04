#include <iostream>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

constexpr size_t ITERS = 10000000;

alignas(64) std::atomic<int64_t> m_bottom{0};
alignas(64) std::atomic<int64_t> m_top{0};

void test_seq_cst_store() {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERS; ++i) {
        m_bottom.store(i, std::memory_order_seq_cst);
        int64_t t = m_top.load(std::memory_order_seq_cst);
        (void)t;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "seq_cst store+load: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
}

void test_xchg() {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERS; ++i) {
        m_bottom.exchange(i, std::memory_order_acq_rel);
        int64_t t = m_top.load(std::memory_order_relaxed);
        (void)t;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "xchg + relaxed load: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
}

void test_store_release_mfence() {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERS; ++i) {
        m_bottom.store(i, std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t t = m_top.load(std::memory_order_relaxed);
        (void)t;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "store release + mfence + relaxed load: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
}

int main() {
    test_seq_cst_store();
    test_xchg();
    test_store_release_mfence();
    return 0;
}
