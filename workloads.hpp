#pragma once
#include <cstdint>
#include <vector>

// High-Compute, Low-Overhead: Dense Matrix Multiplication (O(N^3))
inline void MatrixMultiply(const float* A, const float* B, float* C, int N, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int k = 0; k < N; ++k) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// Low-Compute, High-Overhead: Recursive Fibonacci (O(2^N))
inline uint64_t Fibonacci(uint64_t n) {
    if (n <= 1) return n;
    return Fibonacci(n - 1) + Fibonacci(n - 2);
}
