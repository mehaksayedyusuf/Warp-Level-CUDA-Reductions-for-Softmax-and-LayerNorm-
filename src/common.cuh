#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Standard warp size on NVIDIA architectures
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Full 32-bit lane mask for warp shuffle synchronization
#ifndef FULL_MASK
#define FULL_MASK 0xffffffff
#endif

// Robust CUDA API error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err),              \
                    cudaGetErrorName(err));                                   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Check last kernel launch error and synchronize
#define CUDA_SYNC_CHECK()                                                     \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s (%s)\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err),              \
                    cudaGetErrorName(err));                                   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
        err = cudaDeviceSynchronize();                                        \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA sync error at %s:%d: %s (%s)\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err),              \
                    cudaGetErrorName(err));                                   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// High-precision CUDA Event Timer
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        CUDA_CHECK(cudaEventRecord(start, 0));
    }

    void Stop() {
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
    }

    // Returns elapsed time in milliseconds
    float ElapsedMs() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// Memory bandwidth computation in Gigabytes per second (GB/s)
inline double computeBandwidthGBs(size_t bytesReadWrite, float timeMs) {
    if (timeMs <= 0.0f) return 0.0;
    // bytes / 1e9 / (timeMs / 1e3) = bytes / (timeMs * 1e6)
    return ((double)bytesReadWrite / 1.0e9) / ((double)timeMs / 1000.0);
}

// Check maximum absolute and relative difference between two arrays
inline bool verifyCorrectness(const float* ref, const float* test, size_t n, 
                              float rtol = 1e-4f, float atol = 1e-5f, 
                              float* max_abs_diff = nullptr, float* max_rel_diff = nullptr) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    bool passed = true;

    for (size_t i = 0; i < n; i++) {
        float a = ref[i];
        float b = test[i];
        float abs_diff = fabsf(a - b);
        float rel_diff = abs_diff / (fabsf(a) + 1e-8f);

        if (abs_diff > max_abs) max_abs = abs_diff;
        if (rel_diff > max_rel) max_rel = rel_diff;

        if (isnan(b) || isinf(b) || abs_diff > (atol + rtol * fabsf(a))) {
            if (passed) {
                // Print first failing index for diagnostics
                fprintf(stderr, "Verification mismatch at [%zu]: ref=%.7f, test=%.7f, diff=%.7e\n",
                        i, a, b, abs_diff);
            }
            passed = false;
        }
    }

    if (max_abs_diff) *max_abs_diff = max_abs;
    if (max_rel_diff) *max_rel_diff = max_rel;
    return passed;
}
