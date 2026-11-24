#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <random>

#include "common.cuh"
#include "warp_primitives.cuh"
#include "softmax_kernels.cuh"
#include "layernorm_kernels.cuh"

// CPU Reference Implementations for Correctness Checking
void cpu_softmax(const float* input, float* output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;

        float max_val = -1e30f;
        for (int c = 0; c < cols; c++) {
            if (row_in[c] > max_val) max_val = row_in[c];
        }

        float sum_val = 0.0f;
        for (int c = 0; c < cols; c++) {
            row_out[c] = expf(row_in[c] - max_val);
            sum_val += row_out[c];
        }

        float inv_sum = 1.0f / (sum_val + 1e-12f);
        for (int c = 0; c < cols; c++) {
            row_out[c] *= inv_sum;
        }
    }
}

void cpu_layernorm(const float* input, const float* gamma, const float* beta,
                   float* output, int rows, int cols, float eps = 1e-5f) {
    for (int r = 0; r < rows; r++) {
        const float* row_in = input + r * cols;
        float* row_out = output + r * cols;

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) sum += row_in[c];
        float mean = sum / (float)cols;

        float var_sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float diff = row_in[c] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / (float)cols;
        float inv_std = 1.0f / sqrtf(variance + eps);

        for (int c = 0; c < cols; c++) {
            float g = gamma ? gamma[c] : 1.0f;
            float b = beta  ? beta[c]  : 0.0f;
            row_out[c] = (row_in[c] - mean) * inv_std * g + b;
        }
    }
}

struct BenchmarkRecord {
    std::string operation;
    std::string variant;
    int rows;
    int cols;
    float time_ms;
    double bandwidth_gb_s;
    float speedup;
    bool passed;
    float max_abs_diff;
};

void printBenchmarkTable(const std::vector<BenchmarkRecord>& records) {
    std::cout << "\n========================================================================================================\n";
    std::cout << "                                  CUDA BENCHMARK EVALUATION RESULTS                                     \n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left 
              << std::setw(12) << "Operation"
              << std::setw(30) << "Kernel Variant"
              << std::setw(14) << "Shape (MxK)"
              << std::setw(14) << "Time (ms)"
              << std::setw(16) << "Bandwidth(GB/s)"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "Status"
              << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : records) {
        std::string shape = std::to_string(r.rows) + "x" + std::to_string(r.cols);
        std::cout << std::left
                  << std::setw(12) << r.operation
                  << std::setw(30) << r.variant
                  << std::setw(14) << shape
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << r.time_ms
                  << std::fixed << std::setprecision(2)
                  << std::setw(16) << r.bandwidth_gb_s
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << (std::to_string(r.speedup) + "x")
                  << std::setw(10) << (r.passed ? "PASS" : "FAIL")
                  << "\n";
    }
    std::cout << "========================================================================================================\n\n";
}

int main(int argc, char** argv) {
    std::cout << "=== Warp-Level CUDA Reductions for Softmax and LayerNorm ===\n";

    int rows = 4096;
    int cols = 1024;
    int warmup_iters = 10;
    int bench_iters = 100;

    if (argc >= 3) {
        rows = std::stoi(argv[1]);
        cols = std::stoi(argv[2]);
    }

    std::cout << "Evaluating Matrix Dimensions: Rows = " << rows << ", Cols (Hidden Dim) = " << cols << "\n";
    std::cout << "Total Elements: " << (size_t)rows * cols << " (" << ((double)rows * cols * sizeof(float) / (1024*1024)) << " MB)\n";

    size_t bytes = (size_t)rows * cols * sizeof(float);
    size_t cols_bytes = (size_t)cols * sizeof(float);

    // Host allocations
    std::vector<float> h_in(rows * cols);
    std::vector<float> h_out_ref(rows * cols);
    std::vector<float> h_out_test(rows * cols);
    std::vector<float> h_gamma(cols, 1.0f);
    std::vector<float> h_beta(cols, 0.0f);

    // Random initialization
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    for (size_t i = 0; i < h_in.size(); i++) {
        h_in[i] = dis(gen);
    }
    for (int i = 0; i < cols; i++) {
        h_gamma[i] = 1.0f + 0.1f * dis(gen);
        h_beta[i]  = 0.05f * dis(gen);
    }

    // Device allocations
    float *d_in, *d_out, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, cols_bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cols_bytes, cudaMemcpyHostToDevice));

    std::vector<BenchmarkRecord> records;
    GpuTimer timer;

    // ========================================================================
    // PART 1: SOFTMAX BENCHMARK
    // ========================================================================
    std::cout << "\n>>> Running Softmax Verification & Benchmarks...\n";
    cpu_softmax(h_in.data(), h_out_ref.data(), rows, cols);

    float baseline_softmax_ms = 1.0f;

    // 1.1 Naive Softmax (Shared Memory Tree)
    {
        int threads = (cols <= 1024) ? ((cols + 31) / 32 * 32) : 256;
        size_t shmem = threads * sizeof(float);
        
        for (int i = 0; i < warmup_iters; i++) {
            softmax_naive_kernel<<<rows, threads, shmem>>>(d_in, d_out, rows, cols);
        }
        CUDA_SYNC_CHECK();

        timer.Start();
        for (int i = 0; i < bench_iters; i++) {
            softmax_naive_kernel<<<rows, threads, shmem>>>(d_in, d_out, rows, cols);
        }
        timer.Stop();
        float ms = timer.ElapsedMs() / bench_iters;
        baseline_softmax_ms = ms;

        CUDA_CHECK(cudaMemcpy(h_out_test.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        bool pass = verifyCorrectness(h_out_ref.data(), h_out_test.data(), rows * cols, 1e-4f, 1e-5f, &max_diff);

        // Softmax reads input (bytes) and writes output (bytes)
        double bw = computeBandwidthGBs(2 * bytes, ms);
        records.push_back({"Softmax", "1. Naive (Shared Mem)", rows, cols, ms, bw, 1.0f, pass, max_diff});
    }

    // 1.2 Warp Per Row Softmax (Zero Shmem)
    if (cols <= 2048) {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int blocks = (rows + warps_per_block - 1) / warps_per_block;

        for (int i = 0; i < warmup_iters; i++) {
            softmax_warp_per_row_kernel<<<blocks, threads_per_block>>>(d_in, d_out, rows, cols);
        }
        CUDA_SYNC_CHECK();

        timer.Start();
        for (int i = 0; i < bench_iters; i++) {
            softmax_warp_per_row_kernel<<<blocks, threads_per_block>>>(d_in, d_out, rows, cols);
        }
        timer.Stop();
        float ms = timer.ElapsedMs() / bench_iters;

        CUDA_CHECK(cudaMemcpy(h_out_test.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        bool pass = verifyCorrectness(h_out_ref.data(), h_out_test.data(), rows * cols, 1e-4f, 1e-5f, &max_diff);
        double bw = computeBandwidthGBs(2 * bytes, ms);
        records.push_back({"Softmax", "2. Warp Shuffle (1 Warp/Row)", rows, cols, ms, bw, baseline_softmax_ms / ms, pass, max_diff});
    }

    // 1.3 Online Softmax (FlashAttention Style)
    if (cols <= 2048) {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int blocks = (rows + warps_per_block - 1) / warps_per_block;

        for (int i = 0; i < warmup_iters; i++) {
            softmax_online_warp_kernel<<<blocks, threads_per_block>>>(d_in, d_out, rows, cols);
        }
        CUDA_SYNC_CHECK();

        timer.Start();
        for (int i = 0; i < bench_iters; i++) {
            softmax_online_warp_kernel<<<blocks, threads_per_block>>>(d_in, d_out, rows, cols);
        }
        timer.Stop();
        float ms = timer.ElapsedMs() / bench_iters;

        CUDA_CHECK(cudaMemcpy(h_out_test.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        bool pass = verifyCorrectness(h_out_ref.data(), h_out_test.data(), rows * cols, 1e-4f, 1e-5f, &max_diff);
        double bw = computeBandwidthGBs(2 * bytes, ms);
        records.push_back({"Softmax", "3. Online Warp (1-Pass)", rows, cols, ms, bw, baseline_softmax_ms / ms, pass, max_diff});
    }

    // 1.4 Vectorized Float4 Warp Softmax
    if (cols % 4 == 0) {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int blocks = (rows + warps_per_block - 1) / warps_per_block;

        for (int i = 0; i < warmup_iters; i++) {
            softmax_vectorized_warp_kernel<<<blocks, threads_per_block>>>(d_in, d_out, rows, cols);
        }
        CUDA_SYNC_CHECK();

        timer.Start();
        for (int i = 0; i < bench_iters; i++) {
            softmax_vectorized_warp_kernel<<<blocks, threads_per_block>>>(d_in, d_out, rows, cols);
        }
        timer.Stop();
        float ms = timer.ElapsedMs() / bench_iters;

        CUDA_CHECK(cudaMemcpy(h_out_test.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        bool pass = verifyCorrectness(h_out_ref.data(), h_out_test.data(), rows * cols, 1e-4f, 1e-5f, &max_diff);
        double bw = computeBandwidthGBs(2 * bytes, ms);
        records.push_back({"Softmax", "4. Vectorized Float4 Warp", rows, cols, ms, bw, baseline_softmax_ms / ms, pass, max_diff});
    }

    // ========================================================================
    // PART 2: LAYERNORM BENCHMARK
    // ========================================================================
    std::cout << "\n>>> Running LayerNorm Verification & Benchmarks...\n";
    cpu_layernorm(h_in.data(), h_gamma.data(), h_beta.data(), h_out_ref.data(), rows, cols);

    float baseline_ln_ms = 1.0f;

    // 2.1 Naive LayerNorm (Shared Memory Tree)
    {
        int threads = (cols <= 1024) ? ((cols + 31) / 32 * 32) : 256;
        size_t shmem = threads * sizeof(float);

        for (int i = 0; i < warmup_iters; i++) {
            layernorm_naive_kernel<<<rows, threads, shmem>>>(d_in, d_gamma, d_beta, d_out, rows, cols, 1e-5f);
        }
        CUDA_SYNC_CHECK();

        timer.Start();
        for (int i = 0; i < bench_iters; i++) {
            layernorm_naive_kernel<<<rows, threads, shmem>>>(d_in, d_gamma, d_beta, d_out, rows, cols, 1e-5f);
        }
        timer.Stop();
        float ms = timer.ElapsedMs() / bench_iters;
        baseline_ln_ms = ms;

        CUDA_CHECK(cudaMemcpy(h_out_test.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        bool pass = verifyCorrectness(h_out_ref.data(), h_out_test.data(), rows * cols, 1e-4f, 1e-5f, &max_diff);

        // Reads input, gamma, beta; writes output
        double bw = computeBandwidthGBs(2 * bytes + 2 * cols_bytes, ms);
        records.push_back({"LayerNorm", "1. Naive (Shared Mem)", rows, cols, ms, bw, 1.0f, pass, max_diff});
    }

    // 2.2 Warp-Level Dual Sum LayerNorm (Zero Shmem)
    if (cols <= 2048) {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int blocks = (rows + warps_per_block - 1) / warps_per_block;

        for (int i = 0; i < warmup_iters; i++) {
            layernorm_warp_per_row_kernel<<<blocks, threads_per_block>>>(d_in, d_gamma, d_beta, d_out, rows, cols, 1e-5f);
        }
        CUDA_SYNC_CHECK();

        timer.Start();
        for (int i = 0; i < bench_iters; i++) {
            layernorm_warp_per_row_kernel<<<blocks, threads_per_block>>>(d_in, d_gamma, d_beta, d_out, rows, cols, 1e-5f);
        }
        timer.Stop();
        float ms = timer.ElapsedMs() / bench_iters;

        CUDA_CHECK(cudaMemcpy(h_out_test.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        bool pass = verifyCorrectness(h_out_ref.data(), h_out_test.data(), rows * cols, 1e-4f, 1e-5f, &max_diff);
        double bw = computeBandwidthGBs(2 * bytes + 2 * cols_bytes, ms);
        records.push_back({"LayerNorm", "2. Warp Dual-Sum (1 Warp/Row)", rows, cols, ms, bw, baseline_ln_ms / ms, pass, max_diff});
    }

    // 2.3 Welford Warp LayerNorm
    if (cols <= 2048) {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int blocks = (rows + warps_per_block - 1) / warps_per_block;

        for (int i = 0; i < warmup_iters; i++) {
            layernorm_welford_warp_kernel<<<blocks, threads_per_block>>>(d_in, d_gamma, d_beta, d_out, rows, cols, 1e-5f);
        }
        CUDA_SYNC_CHECK();

        timer.Start();
        for (int i = 0; i < bench_iters; i++) {
            layernorm_welford_warp_kernel<<<blocks, threads_per_block>>>(d_in, d_gamma, d_beta, d_out, rows, cols, 1e-5f);
        }
        timer.Stop();
        float ms = timer.ElapsedMs() / bench_iters;

        CUDA_CHECK(cudaMemcpy(h_out_test.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        bool pass = verifyCorrectness(h_out_ref.data(), h_out_test.data(), rows * cols, 1e-4f, 1e-5f, &max_diff);
        double bw = computeBandwidthGBs(2 * bytes + 2 * cols_bytes, ms);
        records.push_back({"LayerNorm", "3. Welford Online Warp", rows, cols, ms, bw, baseline_ln_ms / ms, pass, max_diff});
    }

    // 2.4 Vectorized Float4 Warp LayerNorm
    if (cols % 4 == 0) {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / WARP_SIZE;
        int blocks = (rows + warps_per_block - 1) / warps_per_block;

        for (int i = 0; i < warmup_iters; i++) {
            layernorm_vectorized_warp_kernel<<<blocks, threads_per_block>>>(d_in, d_gamma, d_beta, d_out, rows, cols, 1e-5f);
        }
        CUDA_SYNC_CHECK();

        timer.Start();
        for (int i = 0; i < bench_iters; i++) {
            layernorm_vectorized_warp_kernel<<<blocks, threads_per_block>>>(d_in, d_gamma, d_beta, d_out, rows, cols, 1e-5f);
        }
        timer.Stop();
        float ms = timer.ElapsedMs() / bench_iters;

        CUDA_CHECK(cudaMemcpy(h_out_test.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        float max_diff = 0.0f;
        bool pass = verifyCorrectness(h_out_ref.data(), h_out_test.data(), rows * cols, 1e-4f, 1e-5f, &max_diff);
        double bw = computeBandwidthGBs(2 * bytes + 2 * cols_bytes, ms);
        records.push_back({"LayerNorm", "4. Vectorized Float4 Warp", rows, cols, ms, bw, baseline_ln_ms / ms, pass, max_diff});
    }

    // Print summary table
    printBenchmarkTable(records);

    // Write JSON file for plotting and reporting
    std::ofstream json_out("results/benchmark_data.json");
    if (json_out.is_open()) {
        json_out << "[\n";
        for (size_t i = 0; i < records.size(); i++) {
            const auto& r = records[i];
            json_out << "  {\n"
                     << "    \"operation\": \"" << r.operation << "\",\n"
                     << "    \"variant\": \"" << r.variant << "\",\n"
                     << "    \"rows\": " << r.rows << ",\n"
                     << "    \"cols\": " << r.cols << ",\n"
                     << "    \"time_ms\": " << r.time_ms << ",\n"
                     << "    \"bandwidth_gb_s\": " << r.bandwidth_gb_s << ",\n"
                     << "    \"speedup\": " << r.speedup << ",\n"
                     << "    \"passed\": " << (r.passed ? "true" : "false") << ",\n"
                     << "    \"max_abs_diff\": " << r.max_abs_diff << "\n"
                     << "  }" << (i + 1 < records.size() ? ",\n" : "\n");
        }
        json_out << "]\n";
        json_out.close();
        std::cout << "Benchmark JSON saved to results/benchmark_data.json\n";
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}
