#pragma once

#include "common.cuh"

/**
 * ============================================================================
 * WARP-LEVEL SHUFFLE REDUCTION PRIMITIVES
 * 
 * Hardware Mechanics:
 * On NVIDIA GPU SMs (Streaming Multiprocessors), 32 threads form a Warp.
 * Prior to Kepler (sm_30), communication between threads required writing to
 * Shared Memory (__shared__) followed by a full thread block barrier (__syncthreads()).
 * 
 * Warp Shuffle intrinsics (__shfl_down_sync, __shfl_xor_sync) allow direct register-to-
 * register data exchange among threads in the same warp within 1-2 clock cycles,
 * with ZERO memory traffic and ZERO shared memory bank conflicts.
 * ============================================================================
 */

// ----------------------------------------------------------------------------
// 1. Warp-Level Reduction Primitives (Single Warp, 32 lanes)
// ----------------------------------------------------------------------------

/**
 * @brief Warp-level parallel reduction for SUM using __shfl_down_sync.
 * Lane 0 receives the total sum across all 32 lanes in 5 instruction steps.
 */
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

/**
 * @brief Warp-level parallel reduction for MAX using __shfl_down_sync.
 * Lane 0 receives the maximum value across all 32 lanes.
 */
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

/**
 * @brief Warp-level ALL-REDUCE for SUM using __shfl_xor_sync (Butterfly reduction).
 * Every thread in the warp receives the final reduced sum without an extra broadcast.
 */
__device__ __forceinline__ float warpAllReduceSum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    }
    return val;
}

/**
 * @brief Warp-level ALL-REDUCE for MAX using __shfl_xor_sync.
 * Every thread in the warp receives the maximum value without extra broadcast.
 */
__device__ __forceinline__ float warpAllReduceMax(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, mask));
    }
    return val;
}

/**
 * @brief Simultaneous dual reduction for SUM(x) and SUM(x^2).
 * Used extensively in LayerNorm to calculate Mean and Variance in lockstep.
 */
__device__ __forceinline__ void warpReduceDualSum(float sum_x, float sum_sq, float &out_sum_x, float &out_sum_sq) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_x  += __shfl_down_sync(FULL_MASK, sum_x,  offset);
        sum_sq += __shfl_down_sync(FULL_MASK, sum_sq, offset);
    }
    out_sum_x  = sum_x;
    out_sum_sq = sum_sq;
}

// ----------------------------------------------------------------------------
// 2. Online & Streaming Warp Reductions
// ----------------------------------------------------------------------------

/**
 * @brief Online Softmax tuple (max, sum_exp) accumulator.
 * Merges two partial Softmax reduction states:
 * m = max(m1, m2)
 * d = d1 * exp(m1 - m) + d2 * exp(m2 - m)
 */
struct OnlineSoftmaxState {
    float m; // running max
    float d; // running sum of exponentials
};

__device__ __forceinline__ OnlineSoftmaxState mergeOnlineSoftmax(OnlineSoftmaxState a, OnlineSoftmaxState b) {
    float new_m = fmaxf(a.m, b.m);
    float new_d = a.d * __expf(a.m - new_m) + b.d * __expf(b.m - new_m);
    return {new_m, new_d};
}

/**
 * @brief Warp-level reduction for Online Softmax (FlashAttention-style).
 * Lane 0 receives the exact globally normalized (max, sum_exp) in 5 cycles.
 */
__device__ __forceinline__ OnlineSoftmaxState warpReduceOnlineSoftmax(OnlineSoftmaxState state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        OnlineSoftmaxState other;
        other.m = __shfl_down_sync(FULL_MASK, state.m, offset);
        other.d = __shfl_down_sync(FULL_MASK, state.d, offset);
        state = mergeOnlineSoftmax(state, other);
    }
    return state;
}

/**
 * @brief Online Welford state for single-pass numerically stable Mean & Variance.
 * Count n, mean mu, and sum of squared differences M2 = sum((x - mu)^2).
 */
struct WelfordState {
    float n;
    float mean;
    float m2;
};

__device__ __forceinline__ WelfordState mergeWelford(WelfordState a, WelfordState b) {
    if (a.n <= 0.0f) return b;
    if (b.n <= 0.0f) return a;
    float total_n = a.n + b.n;
    float delta = b.mean - a.mean;
    float new_mean = a.mean + delta * (b.n / total_n);
    float new_m2 = a.m2 + b.m2 + delta * delta * (a.n * b.n / total_n);
    return {total_n, new_mean, new_m2};
}

__device__ __forceinline__ WelfordState warpReduceWelford(WelfordState state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        WelfordState other;
        other.n    = __shfl_down_sync(FULL_MASK, state.n,    offset);
        other.mean = __shfl_down_sync(FULL_MASK, state.mean, offset);
        other.m2   = __shfl_down_sync(FULL_MASK, state.m2,   offset);
        state = mergeWelford(state, other);
    }
    return state;
}

// ----------------------------------------------------------------------------
// 3. Block-Level Reductions utilizing Warp Shuffles + Shared Memory Buffer
// ----------------------------------------------------------------------------

/**
 * @brief High-performance Block-Level SUM Reduction.
 * Combines warp-level shuffles with a minimal shared memory buffer (max 32 floats).
 * 
 * Step 1: Every warp reduces internally via warpReduceSum (register-only).
 * Step 2: Lane 0 of each warp stores its scalar result into shared_mem[warp_id].
 * Step 3: Thread block synchronizes (__syncthreads()).
 * Step 4: The first warp (threads 0 .. num_warps - 1) loads and performs a 
 *         final warpReduceSum to obtain the block-wide reduction scalar.
 * Step 5: Broadcast result back to shared_mem[0].
 */
__device__ __forceinline__ float blockReduceSum(float val, float* shared_mem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // 1. Intra-warp reduction
    val = warpReduceSum(val);

    // 2. Write warp results to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    // 3. Final reduction by the first warp
    if (warp_id == 0) {
        float warp_val = (lane_id < num_warps) ? shared_mem[lane_id] : 0.0f;
        warp_val = warpReduceSum(warp_val);
        if (lane_id == 0) {
            shared_mem[0] = warp_val;
        }
    }
    __syncthreads();

    return shared_mem[0];
}

/**
 * @brief High-performance Block-Level MAX Reduction.
 */
__device__ __forceinline__ float blockReduceMax(float val, float* shared_mem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    val = warpReduceMax(val);

    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float warp_val = (lane_id < num_warps) ? shared_mem[lane_id] : -1e30f;
        warp_val = warpReduceMax(warp_val);
        if (lane_id == 0) {
            shared_mem[0] = warp_val;
        }
    }
    __syncthreads();

    return shared_mem[0];
}
