#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace pgc_eval {

// ----------------------------------------------------------------------------
// Mathematical Reference for Softmax
// ----------------------------------------------------------------------------

// 1. Naive Unsafe Softmax (Demonstrates FP32 overflow when x > 88.0)
inline void softmax_naive_cpu(const float* in, float* out, int D) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        out[i] = std::exp(in[i]);
        sum += out[i];
    }
    float inv_sum = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < D; i++) {
        out[i] *= inv_sum;
    }
}

// 2. Safe Softmax (Standard 3-pass: max subtraction, exp + sum, normalize)
inline void softmax_safe_cpu(const float* in, float* out, int D) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < D; i++) {
        if (in[i] > max_val) max_val = in[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        out[i] = std::exp(in[i] - max_val);
        sum += out[i];
    }

    float inv_sum = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < D; i++) {
        out[i] *= inv_sum;
    }
}

// 3. Online Softmax (FlashAttention-style single streaming pass)
inline void softmax_online_cpu(const float* in, float* out, int D) {
    float m = -std::numeric_limits<float>::infinity();
    float d = 0.0f;

    for (int i = 0; i < D; i++) {
        float x = in[i];
        float m_prev = m;
        m = std::max(m_prev, x);
        d = d * std::exp(m_prev - m) + std::exp(x - m);
    }

    float inv_sum = 1.0f / (d + 1e-12f);
    for (int i = 0; i < D; i++) {
        out[i] = std::exp(in[i] - m) * inv_sum;
    }
}

// ----------------------------------------------------------------------------
// Mathematical Reference for LayerNorm
// ----------------------------------------------------------------------------

// 1. Standard Two-Pass LayerNorm
inline void layernorm_twopass_cpu(const float* in, const float* gamma, const float* beta,
                                  float* out, int D, float eps = 1e-5f) {
    float sum = 0.0f;
    for (int i = 0; i < D; i++) sum += in[i];
    float mean = sum / (float)D;

    float sum_sq = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = in[i] - mean;
        sum_sq += diff * diff;
    }
    float variance = sum_sq / (float)D;
    float inv_std = 1.0f / std::sqrt(variance + eps);

    for (int i = 0; i < D; i++) {
        float g = gamma ? gamma[i] : 1.0f;
        float b = beta  ? beta[i]  : 0.0f;
        out[i] = (in[i] - mean) * inv_std * g + b;
    }
}

// 2. Dual-Sum LayerNorm (Single loop for sum(x) and sum(x^2))
inline void layernorm_dualsum_cpu(const float* in, const float* gamma, const float* beta,
                                  float* out, int D, float eps = 1e-5f) {
    float sum_x = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < D; i++) {
        float x = in[i];
        sum_x  += x;
        sum_sq += x * x;
    }

    float mean = sum_x / (float)D;
    float variance = std::max(0.0f, (sum_sq / (float)D) - (mean * mean));
    float inv_std = 1.0f / std::sqrt(variance + eps);

    for (int i = 0; i < D; i++) {
        float g = gamma ? gamma[i] : 1.0f;
        float b = beta  ? beta[i]  : 0.0f;
        out[i] = (in[i] - mean) * inv_std * g + b;
    }
}

// 3. Online Welford LayerNorm (Numerically Stable Variance in a single pass)
inline void layernorm_welford_cpu(const float* in, const float* gamma, const float* beta,
                                  float* out, int D, float eps = 1e-5f) {
    float mean = 0.0f;
    float M2 = 0.0f;
    for (int i = 0; i < D; i++) {
        float x = in[i];
        float delta = x - mean;
        mean += delta / (float)(i + 1);
        float delta2 = x - mean;
        M2 += delta * delta2;
    }

    float variance = std::max(0.0f, M2 / (float)D);
    float inv_std = 1.0f / std::sqrt(variance + eps);

    for (int i = 0; i < D; i++) {
        float g = gamma ? gamma[i] : 1.0f;
        float b = beta  ? beta[i]  : 0.0f;
        out[i] = (in[i] - mean) * inv_std * g + b;
    }
}

// ----------------------------------------------------------------------------
// Warp-Level Shuffle Simulation on CPU (Simulates 32-lane SIMT execution)
// ----------------------------------------------------------------------------
inline float simulate_warp_reduce_sum(const float warp_registers[32]) {
    float regs[32];
    for (int i = 0; i < 32; i++) regs[i] = warp_registers[i];

    // Butterfly / tree reduction via shfl_down emulation
    for (int offset = 16; offset > 0; offset /= 2) {
        for (int lane = 0; lane < 32; lane++) {
            if (lane + offset < 32) {
                regs[lane] += regs[lane + offset];
            }
        }
    }
    return regs[0]; // Lane 0 holds total sum
}

inline float simulate_warp_reduce_max(const float warp_registers[32]) {
    float regs[32];
    for (int i = 0; i < 32; i++) regs[i] = warp_registers[i];

    for (int offset = 16; offset > 0; offset /= 2) {
        for (int lane = 0; lane < 32; lane++) {
            if (lane + offset < 32) {
                regs[lane] = std::max(regs[lane], regs[lane + offset]);
            }
        }
    }
    return regs[0];
}

} // namespace pgc_eval
