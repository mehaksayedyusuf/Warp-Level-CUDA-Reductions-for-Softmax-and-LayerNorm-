#pragma once

#include "common.cuh"
#include "warp_primitives.cuh"

/**
 * ============================================================================
 * SOFTMAX KERNELS: NAIVE (SHARED MEMORY) VS. WARP-LEVEL SHUFFLE OPTIMIZATIONS
 * 
 * Mathematical Formulation:
 * For row x of length D:
 * 1. Safe max subtraction:   m = max_{j=0..D-1}(x_j)
 * 2. Exponential & Sum:      d = sum_{j=0..D-1} exp(x_j - m)
 * 3. Softmax Normalization:  y_i = exp(x_i - m) / d
 * ============================================================================
 */

// ----------------------------------------------------------------------------
// 1. NAIVE SOFTMAX: Classical Shared-Memory Tree Reduction
// ----------------------------------------------------------------------------
__global__ void softmax_naive_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Phase 1: Local Max Reduction across thread's assigned elements
    float local_max = -1e30f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Traditional Tree Reduction in Shared Memory with barriers
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Phase 2: Compute exponentials and sum reduction in Shared Memory
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += __expf(row_in[col] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // Phase 3: Final normalization and write to global memory
    float inv_sum = 1.0f / (row_sum + 1e-12f);
    for (int col = tid; col < cols; col += blockDim.x) {
        row_out[col] = __expf(row_in[col] - row_max) * inv_sum;
    }
}

// ----------------------------------------------------------------------------
// 2. WARP-LEVEL SOFTMAX (One Warp Per Row - Highly Efficient for cols <= 1024)
// Zero shared memory used, zero __syncthreads(), register-only communication.
// ----------------------------------------------------------------------------
__global__ void softmax_warp_per_row_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int rows, int cols) {
    // Each warp processes one complete row
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = input + warp_id * cols;
    float* row_out = output + warp_id * cols;

    // Step 1: Thread-local max
    float local_max = -1e30f;
    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    // Warp all-reduce max: all 32 lanes get row_max without broadcast overhead
    float row_max = warpAllReduceMax(local_max);

    // Step 2: Thread-local sum of exponentials
    float local_sum = 0.0f;
    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        local_sum += __expf(row_in[col] - row_max);
    }
    // Warp all-reduce sum: all 32 lanes get row_sum
    float row_sum = warpAllReduceSum(local_sum);

    // Step 3: Write normalized output
    float inv_sum = 1.0f / (row_sum + 1e-12f);
    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        row_out[col] = __expf(row_in[col] - row_max) * inv_sum;
    }
}

// ----------------------------------------------------------------------------
// 3. WARP-OPTIMIZED BLOCK SOFTMAX (One Block Per Row - Scales to large cols)
// Warps reduce in registers, exchange scalars through tiny shared memory (<= 32 elements).
// ----------------------------------------------------------------------------
__global__ void softmax_block_warp_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float s_mem[32]; // Accommodates up to 1024 threads (32 warps)
    int tid = threadIdx.x;
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // 1. Warp + Block Max Reduction
    float local_max = -1e30f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    float row_max = blockReduceMax(local_max, s_mem);

    // 2. Warp + Block Sum Reduction
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += __expf(row_in[col] - row_max);
    }
    float row_sum = blockReduceSum(local_sum, s_mem);

    // 3. Normalized output
    float inv_sum = 1.0f / (row_sum + 1e-12f);
    for (int col = tid; col < cols; col += blockDim.x) {
        row_out[col] = __expf(row_in[col] - row_max) * inv_sum;
    }
}

// ----------------------------------------------------------------------------
// 4. ONLINE SOFTMAX WITH WARP REDUCTIONS (FlashAttention-Style Single Pass)
// Fuses max and sum reduction into a single streaming loop.
// ----------------------------------------------------------------------------
__global__ void softmax_online_warp_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int rows, int cols) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = input + warp_id * cols;
    float* row_out = output + warp_id * cols;

    // Single streaming pass: dynamically update running max & sum
    OnlineSoftmaxState state = {-1e30f, 0.0f};
    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        float val = row_in[col];
        state = mergeOnlineSoftmax(state, {val, 1.0f});
    }

    // Warp-level reduction of the OnlineSoftmax tuple (m, d)
    state = warpReduceOnlineSoftmax(state);

    // Broadcast reduced (m, d) from lane 0 to all lanes
    float global_m = __shfl_sync(FULL_MASK, state.m, 0);
    float global_d = __shfl_sync(FULL_MASK, state.d, 0);

    float inv_sum = 1.0f / (global_d + 1e-12f);
    for (int col = lane_id; col < cols; col += WARP_SIZE) {
        row_out[col] = __expf(row_in[col] - global_m) * inv_sum;
    }
}

// ----------------------------------------------------------------------------
// 5. VECTORIZED FLOAT4 WARP SOFTMAX (Saturating 128-bit Memory Bus)
// Loads 4 floats at once per thread to saturate PCIe and GPU memory controllers.
// ----------------------------------------------------------------------------
__global__ void softmax_vectorized_warp_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               int rows, int cols) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = input + warp_id * cols;
    float* row_out = output + warp_id * cols;
    int cols_vec = cols / 4;

    const float4* vec_in = reinterpret_cast<const float4*>(row_in);
    float4* vec_out = reinterpret_cast<float4*>(row_out);

    // Step 1: Vectorized Max Reduction
    float local_max = -1e30f;
    for (int c = lane_id; c < cols_vec; c += WARP_SIZE) {
        float4 v = vec_in[c];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    // Handle remaining elements if cols % 4 != 0
    for (int c = cols_vec * 4 + lane_id; c < cols; c += WARP_SIZE) {
        local_max = fmaxf(local_max, row_in[c]);
    }
    float row_max = warpAllReduceMax(local_max);

    // Step 2: Vectorized Exponentiation & Sum Reduction
    float local_sum = 0.0f;
    for (int c = lane_id; c < cols_vec; c += WARP_SIZE) {
        float4 v = vec_in[c];
        local_sum += __expf(v.x - row_max) + __expf(v.y - row_max) +
                     __expf(v.z - row_max) + __expf(v.w - row_max);
    }
    for (int c = cols_vec * 4 + lane_id; c < cols; c += WARP_SIZE) {
        local_sum += __expf(row_in[c] - row_max);
    }
    float row_sum = warpAllReduceSum(local_sum);

    // Step 3: Vectorized Write Back
    float inv_sum = 1.0f / (row_sum + 1e-12f);
    for (int c = lane_id; c < cols_vec; c += WARP_SIZE) {
        float4 v = vec_in[c];
        float4 out_v;
        out_v.x = __expf(v.x - row_max) * inv_sum;
        out_v.y = __expf(v.y - row_max) * inv_sum;
        out_v.z = __expf(v.z - row_max) * inv_sum;
        out_v.w = __expf(v.w - row_max) * inv_sum;
        vec_out[c] = out_v;
    }
    for (int c = cols_vec * 4 + lane_id; c < cols; c += WARP_SIZE) {
        row_out[c] = __expf(row_in[c] - row_max) * inv_sum;
    }
}
