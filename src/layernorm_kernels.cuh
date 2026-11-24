#pragma once

#include "common.cuh"
#include "warp_primitives.cuh"

/**
 * ============================================================================
 * LAYERNORM KERNELS: NAIVE (SHARED MEMORY) VS. WARP-LEVEL SHUFFLE OPTIMIZATIONS
 * 
 * Mathematical Formulation:
 * For row x of length D, learnable weights gamma, and bias beta:
 * 1. Mean:                 mu = (1 / D) * sum_{j=0..D-1} x_j
 * 2. Variance:             sigma^2 = (1 / D) * sum_{j=0..D-1} (x_j - mu)^2
 * 3. Standard Deviation:   inv_std = 1 / sqrt(sigma^2 + epsilon)
 * 4. Affine Transform:     y_i = (x_i - mu) * inv_std * gamma_i + beta_i
 * ============================================================================
 */

// ----------------------------------------------------------------------------
// 1. NAIVE LAYERNORM: Classical Shared-Memory Tree Reduction
// ----------------------------------------------------------------------------
__global__ void layernorm_naive_kernel(const float* __restrict__ input,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ output,
                                       int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Phase 1: Sum reduction for Mean
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += row_in[col];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float mean = sdata[0] / (float)cols;
    __syncthreads();

    // Phase 2: Sum of squared differences for Variance
    float local_var_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float diff = row_in[col] - mean;
        local_var_sum += diff * diff;
    }
    sdata[tid] = local_var_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float variance = sdata[0] / (float)cols;
    float inv_std = rsqrtf(variance + eps);
    __syncthreads();

    // Phase 3: Normalize, scale (gamma), and shift (beta)
    for (int col = tid; col < cols; col += blockDim.x) {
        float g = (gamma != nullptr) ? gamma[col] : 1.0f;
        float b = (beta != nullptr)  ? beta[col]  : 0.0f;
        row_out[col] = (row_in[col] - mean) * inv_std * g + b;
    }
}

// ----------------------------------------------------------------------------
// 2. WARP-LEVEL LAYERNORM: Dual Sum Reduction in Registers
// Computes sum(x) and sum(x^2) in parallel with zero shared memory and zero barriers.
// ----------------------------------------------------------------------------
__global__ void layernorm_warp_per_row_kernel(const float* __restrict__ input,
                                             const float* __restrict__ gamma,
                                             const float* __restrict__ beta,
                                             float* __restrict__ output,
                                             int rows, int cols, float eps) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = input + warp_id * cols;
    float* row_out = output + warp_id * cols;

    // Step 1: Accumulate local sum(x) and sum(x^2)
    float local_x = 0.0f;
    float local_sq = 0.0f;
    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        float val = row_in[col];
        local_x  += val;
        local_sq += val * val;
    }

    // Step 2: Dual warp reduction in parallel within 5 cycles
    float sum_x, sum_sq;
    warpReduceDualSum(local_x, local_sq, sum_x, sum_sq);

    // Step 3: Broadcast mean and inv_std to all 32 lanes
    float global_sum_x  = __shfl_sync(FULL_MASK, sum_x,  0);
    float global_sum_sq = __shfl_sync(FULL_MASK, sum_sq, 0);

    float mean = global_sum_x / (float)cols;
    float variance = fmaxf(0.0f, (global_sum_sq / (float)cols) - (mean * mean));
    float inv_std = rsqrtf(variance + eps);

    // Step 4: Normalize, scale, and shift directly in registers
    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        float g = (gamma != nullptr) ? gamma[col] : 1.0f;
        float b = (beta != nullptr)  ? beta[col]  : 0.0f;
        row_out[col] = (row_in[col] - mean) * inv_std * g + b;
    }
}

// ----------------------------------------------------------------------------
// 3. ONLINE WELFORD WARP LAYERNORM (Single Pass, Maximum Numerical Stability)
// Prevents catastrophic cancellation (loss of significance) in variance calculation.
// ----------------------------------------------------------------------------
__global__ void layernorm_welford_warp_kernel(const float* __restrict__ input,
                                             const float* __restrict__ gamma,
                                             const float* __restrict__ beta,
                                             float* __restrict__ output,
                                             int rows, int cols, float eps) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = input + warp_id * cols;
    float* row_out = output + warp_id * cols;

    // Local Welford accumulation
    WelfordState state = {0.0f, 0.0f, 0.0f};
    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        float val = row_in[col];
        state = mergeWelford(state, {1.0f, val, 0.0f});
    }

    // Warp reduction of Welford tuples (n, mean, M2)
    state = warpReduceWelford(state);

    float mean = __shfl_sync(FULL_MASK, state.mean, 0);
    float m2   = __shfl_sync(FULL_MASK, state.m2,   0);
    float variance = fmaxf(0.0f, m2 / (float)cols);
    float inv_std = rsqrtf(variance + eps);

    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        float g = (gamma != nullptr) ? gamma[col] : 1.0f;
        float b = (beta != nullptr)  ? beta[col]  : 0.0f;
        row_out[col] = (row_in[col] - mean) * inv_std * g + b;
    }
}

// ----------------------------------------------------------------------------
// 4. VECTORIZED FLOAT4 WARP LAYERNORM (128-bit Memory Transactions)
// ----------------------------------------------------------------------------
__global__ void layernorm_vectorized_warp_kernel(const float* __restrict__ input,
                                                 const float* __restrict__ gamma,
                                                 const float* __restrict__ beta,
                                                 float* __restrict__ output,
                                                 int rows, int cols, float eps) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = input + warp_id * cols;
    float* row_out = output + warp_id * cols;
    int cols_vec = cols / 4;

    const float4* vec_in = reinterpret_cast<const float4*>(row_in);
    float4* vec_out = reinterpret_cast<float4*>(row_out);
    const float4* vec_gamma = reinterpret_cast<const float4*>(gamma);
    const float4* vec_beta  = reinterpret_cast<const float4*>(beta);

    // Vectorized dual sum
    float local_x = 0.0f;
    float local_sq = 0.0f;
    for (int c = lane_id; c < cols_vec; c += WARP_SIZE) {
        float4 v = vec_in[c];
        local_x  += v.x + v.y + v.z + v.w;
        local_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int c = cols_vec * 4 + lane_id; c < cols; c += WARP_SIZE) {
        float val = row_in[c];
        local_x  += val;
        local_sq += val * val;
    }

    float sum_x, sum_sq;
    warpReduceDualSum(local_x, local_sq, sum_x, sum_sq);

    float global_sum_x  = __shfl_sync(FULL_MASK, sum_x,  0);
    float global_sum_sq = __shfl_sync(FULL_MASK, sum_sq, 0);

    float mean = global_sum_x / (float)cols;
    float variance = fmaxf(0.0f, (global_sum_sq / (float)cols) - (mean * mean));
    float inv_std = rsqrtf(variance + eps);

    // Vectorized write back
    for (int c = lane_id; c < cols_vec; c += WARP_SIZE) {
        float4 v = vec_in[c];
        float4 g = (vec_gamma != nullptr) ? vec_gamma[c] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 b = (vec_beta  != nullptr) ? vec_beta[c]  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 out_v;
        out_v.x = (v.x - mean) * inv_std * g.x + b.x;
        out_v.y = (v.y - mean) * inv_std * g.y + b.y;
        out_v.z = (v.z - mean) * inv_std * g.z + b.z;
        out_v.w = (v.w - mean) * inv_std * g.w + b.w;
        vec_out[c] = out_v;
    }
    for (int c = cols_vec * 4 + lane_id; c < cols; c += WARP_SIZE) {
        float g = (gamma != nullptr) ? gamma[c] : 1.0f;
        float b = (beta  != nullptr) ? beta[c]  : 0.0f;
        row_out[c] = (row_in[c] - mean) * inv_std * g + b;
    }
}
