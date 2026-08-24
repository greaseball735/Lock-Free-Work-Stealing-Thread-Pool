#pragma once
#include <vector>
#include <cstdint>
#include <complex>
#include <atomic>
#include "omp.h"

// =============================================================================
// WORKLOAD 3: ASYMMETRIC QUICKSORT
// =============================================================================

inline int Partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return (i + 1);
}

inline void QsSequential(int* arr, int low, int high, int cutoff = 128) {
    if (low < high) {
        if (high - low < cutoff) {
            // Simple insertion sort for small partitions to eliminate overhead
            for (int i = low + 1; i <= high; i++) {
                int key = arr[i];
                int j = i - 1;
                while (j >= low && arr[j] > key) {
                    arr[j + 1] = arr[j];
                    j = j - 1;
                }
                arr[j + 1] = key;
            }
            return;
        }
        int pi = Partition(arr, low, high);
        QsSequential(arr, low, pi - 1, cutoff);
        QsSequential(arr, pi + 1, high, cutoff);
    }
}

inline void QsOMP(int* arr, int low, int high, int cutoff) {
    if (low < high) {
        if (high - low < cutoff) {
            QsSequential(arr, low, high, cutoff);
            return;
        }
        int pi = Partition(arr, low, high);
        
        #pragma omp task shared(arr) firstprivate(low, pi, cutoff)
        QsOMP(arr, low, pi - 1, cutoff);
        
        #pragma omp task shared(arr) firstprivate(high, pi, cutoff)
        QsOMP(arr, pi + 1, high, cutoff);
        
        #pragma omp taskwait
    }
}

// =============================================================================
// WORKLOAD 4: MANDELBROT SET (DYNAMIC GRID)
// =============================================================================

inline void ComputeMandelbrotChunk(int* output, int width, int height, 
                                   int start_x, int end_x, int start_y, int end_y, int max_iter) {
    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            float cr = (x - width / 2.0f) * 4.0f / width;
            float ci = (y - height / 2.0f) * 4.0f / height;
            float zr = 0.0f, zi = 0.0f;
            int iter = 0;
            
            while (zr * zr + zi * zi <= 4.0f && iter < max_iter) {
                float temp = zr * zr - zi * zi + cr;
                zi = 2.0f * zr * zi + ci;
                zr = temp;
                iter++;
            }
            output[y * width + x] = iter;
        }
    }
}
