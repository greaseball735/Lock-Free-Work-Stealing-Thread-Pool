#include "workloads.hpp"
#include "naive_thread_pool.hpp"
#include <iostream>
#include <chrono>
#include <omp.h>

void BenchmarkMatrixMultiplication() {
    constexpr int N = 1024; // 1024x1024 matrix
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);
    
    int num_threads = std::thread::hardware_concurrency();
    int chunk_size = N / num_threads;

    // --- Baseline 0: Sequential ---
    auto start_seq = std::chrono::high_resolution_clock::now();
    MatrixMultiply(A.data(), B.data(), C.data(), N, 0, N);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::cout << "Baseline 0 (Sequential) Matrix: " 
              << std::chrono::duration<double, std::milli>(end_seq - start_seq).count() << " ms\n";

    // Clear C array
    std::fill(C.begin(), C.end(), 0.0f);

    // --- Baseline 1: Naive Thread Pool ---
    NaiveThreadPool pool(num_threads);
    auto start_pool = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * chunk_size;
        int end_row = (i == num_threads - 1) ? N : start_row + chunk_size;
        pool.Enqueue([&A, &B, &C, N, start_row, end_row]() {
            MatrixMultiply(A.data(), B.data(), C.data(), N, start_row, end_row);
        });
    }
    pool.WaitAll();
    auto end_pool = std::chrono::high_resolution_clock::now();
    std::cout << "Baseline 1 (Mutex Pool) Matrix: " 
              << std::chrono::duration<double, std::milli>(end_pool - start_pool).count() << " ms\n";

    // Clear C array
    std::fill(C.begin(), C.end(), 0.0f);

    // --- Baseline 2: OpenMP ---
    auto start_omp = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static, chunk_size)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
    auto end_omp = std::chrono::high_resolution_clock::now();
    std::cout << "Baseline 2 (OpenMP) Matrix: " 
              << std::chrono::duration<double, std::milli>(end_omp - start_omp).count() << " ms\n";
}

int main() {
    std::cout << "Starting Benchmarks...\n";
    BenchmarkMatrixMultiplication();
    return 0;
}
