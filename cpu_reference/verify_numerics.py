import math
import json
import numpy as np

def simulate_warp_reduce_sum(lanes: np.ndarray) -> float:
    """Simulates 32-thread SIMT butterfly warp reduction via __shfl_down_sync."""
    assert len(lanes) == 32
    regs = lanes.astype(np.float32).copy()
    offset = 16
    while offset > 0:
        for lane in range(32):
            if lane + offset < 32:
                regs[lane] += regs[lane + offset]
        offset //= 2
    return float(regs[0])

def simulate_warp_reduce_max(lanes: np.ndarray) -> float:
    """Simulates 32-thread SIMT warp reduction for MAX via __shfl_down_sync."""
    assert len(lanes) == 32
    regs = lanes.astype(np.float32).copy()
    offset = 16
    while offset > 0:
        for lane in range(32):
            if lane + offset < 32:
                regs[lane] = max(regs[lane], regs[lane + offset])
        offset //= 2
    return float(regs[0])

def softmax_safe(x: np.ndarray) -> np.ndarray:
    """Safe Softmax with max subtraction."""
    x = x.astype(np.float32)
    m = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - m)
    d = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / d

def softmax_online(x: np.ndarray) -> np.ndarray:
    """FlashAttention / Online Softmax single streaming pass."""
    x = x.astype(np.float32)
    rows, cols = x.shape
    out = np.empty_like(x)
    for r in range(rows):
        m = -float('inf')
        d = 0.0
        for c in range(cols):
            val = x[r, c]
            m_prev = m
            m = max(m_prev, val)
            d = d * math.exp(m_prev - m) + math.exp(val - m)
        out[r, :] = np.exp(x[r, :] - m) / d
    return out

def layernorm_twopass(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * gamma + beta

def layernorm_dualsum(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float32)
    cols = x.shape[-1]
    sum_x = np.sum(x, axis=-1, keepdims=True)
    sum_sq = np.sum(x ** 2, axis=-1, keepdims=True)
    mean = sum_x / cols
    var = np.maximum(0.0, (sum_sq / cols) - (mean ** 2))
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * gamma + beta

def layernorm_welford(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = x.astype(np.float32)
    rows, cols = x.shape
    out = np.empty_like(x)
    for r in range(rows):
        mean = 0.0
        M2 = 0.0
        for c in range(cols):
            val = float(x[r, c])
            delta = val - mean
            mean += delta / (c + 1)
            delta2 = val - mean
            M2 += delta * delta2
        var = max(0.0, M2 / cols)
        inv_std = 1.0 / math.sqrt(var + eps)
        out[r, :] = (x[r, :] - mean) * inv_std * gamma + beta
    return out

def run_all_tests():
    print("=" * 80)
    print("   PGC EVALUATION: NUMERICAL & SIMT VERIFICATION SUITE")
    print("=" * 80)

    np.random.seed(42)
    results = {}

    # Test 1: Warp Reduction Simulation (32 threads)
    test_lanes = np.random.uniform(-5.0, 5.0, size=32).astype(np.float32)
    seq_sum = float(np.sum(test_lanes))
    seq_max = float(np.max(test_lanes))
    warp_sum = simulate_warp_reduce_sum(test_lanes)
    warp_max = simulate_warp_reduce_max(test_lanes)

    diff_sum = abs(seq_sum - warp_sum)
    diff_max = abs(seq_max - warp_max)
    print(f"\n[TEST 1] 32-Lane Warp Butterfly Reduction Simulation:")
    print(f"  - Sequential Sum: {seq_sum:.6f} | Warp Reduced Sum: {warp_sum:.6f} (Diff: {diff_sum:.2e})")
    print(f"  - Sequential Max: {seq_max:.6f} | Warp Reduced Max: {warp_max:.6f} (Diff: {diff_max:.2e})")
    t1_pass = (diff_sum < 1e-4) and (diff_max < 1e-6)
    print(f"  -> Result: {'PASS' if t1_pass else 'FAIL'}")
    results['test1_warp_shuffle'] = {'passed': t1_pass, 'diff_sum': diff_sum, 'diff_max': diff_max}

    # Test 2: Softmax Equivalence (Safe vs Online)
    rows, cols = 128, 512
    x_test = np.random.normal(0, 2.0, size=(rows, cols)).astype(np.float32)
    out_safe = softmax_safe(x_test)
    out_online = softmax_online(x_test)
    max_sm_diff = float(np.max(np.abs(out_safe - out_online)))
    sum_probs_err = float(np.max(np.abs(np.sum(out_online, axis=-1) - 1.0)))
    print(f"\n[TEST 2] Softmax Safe vs. Online (FlashAttention) Equivalence ({rows}x{cols}):")
    print(f"  - Maximum Absolute Output Difference: {max_sm_diff:.2e}")
    print(f"  - Conservation of Probability Sum Error: {sum_probs_err:.2e}")
    t2_pass = (max_sm_diff < 1e-6) and (sum_probs_err < 1e-5)
    print(f"  -> Result: {'PASS' if t2_pass else 'FAIL'}")
    results['test2_softmax_equivalence'] = {'passed': t2_pass, 'max_diff': max_sm_diff}

    # Test 3: Softmax FP32 Overflow Resistance
    x_overflow = np.array([[100.0, 102.5, 105.0]], dtype=np.float32)
    out_safe_of = softmax_safe(x_overflow)
    out_online_of = softmax_online(x_overflow)
    # Naive without max subtraction exp(100) overflows FP32
    exp_naive = np.exp(x_overflow)
    naive_overflows = np.isinf(exp_naive).any()
    safe_valid = not np.isnan(out_safe_of).any() and not np.isinf(out_safe_of).any()
    print(f"\n[TEST 3] Softmax Extreme Value Stress Test (x in [100.0, 105.0]):")
    print(f"  - Naive exp(105.0) FP32 Overflow: {'YES (Inf detected)' if naive_overflows else 'NO'}")
    print(f"  - Safe Softmax Output: {out_safe_of[0]}")
    print(f"  - Online Softmax Output: {out_online_of[0]}")
    t3_pass = naive_overflows and safe_valid
    print(f"  -> Result: {'PASS' if t3_pass else 'FAIL'}")
    results['test3_overflow_resistance'] = {'passed': t3_pass}

    # Test 4: LayerNorm Equivalence (Two-Pass vs Dual-Sum vs Welford)
    gamma = np.random.uniform(0.8, 1.2, size=cols).astype(np.float32)
    beta = np.random.uniform(-0.1, 0.1, size=cols).astype(np.float32)
    ln_two = layernorm_twopass(x_test, gamma, beta)
    ln_dual = layernorm_dualsum(x_test, gamma, beta)
    ln_welford = layernorm_welford(x_test, gamma, beta)

    diff_dual = float(np.max(np.abs(ln_two - ln_dual)))
    diff_welford = float(np.max(np.abs(ln_two - ln_welford)))
    print(f"\n[TEST 4] LayerNorm Mathematical Equivalence ({rows}x{cols}):")
    print(f"  - Max Diff (Two-Pass vs Dual-Sum): {diff_dual:.2e}")
    print(f"  - Max Diff (Two-Pass vs Welford):  {diff_welford:.2e}")
    t4_pass = (diff_dual < 1e-4) and (diff_welford < 1e-4)
    print(f"  -> Result: {'PASS' if t4_pass else 'FAIL'}")
    results['test4_layernorm_equivalence'] = {'passed': t4_pass, 'diff_dual': diff_dual, 'diff_welford': diff_welford}

    # Test 5: LayerNorm Catastrophic Cancellation Contrast
    # In Dual-Sum: sum(x^2)/D - mean^2 computes difference of two large numbers.
    # When inputs have large mean (e.g. 1000.0) relative to variance, Dual-Sum loses precision.
    # Welford computes deviations directly and is stable.
    x_shifted = (1000.0 + np.random.normal(0, 1.0, size=(16, 256))).astype(np.float32)
    ln_two_shift = layernorm_twopass(x_shifted, np.ones(256, dtype=np.float32), np.zeros(256, dtype=np.float32))
    ln_welf_shift = layernorm_welford(x_shifted, np.ones(256, dtype=np.float32), np.zeros(256, dtype=np.float32))
    
    welf_mean_check = float(np.max(np.abs(np.mean(ln_welf_shift, axis=-1))))
    welf_var_check = float(np.max(np.abs(np.var(ln_welf_shift, axis=-1) - 1.0)))
    print(f"\n[TEST 5] LayerNorm Large Shift Stability Test (Input ~ 1,000.0 + N(0,1)):")
    print(f"  - Welford Normalized Mean Error: {welf_mean_check:.2e}")
    print(f"  - Welford Normalized Variance Error: {welf_var_check:.2e}")
    t5_pass = (welf_mean_check < 1e-4) and (welf_var_check < 1e-3)
    print(f"  -> Result: {'PASS' if t5_pass else 'FAIL'}")
    results['test5_cancellation_resistance'] = {'passed': t5_pass}

    all_passed = t1_pass and t2_pass and t3_pass and t4_pass and t5_pass
    print("\n" + "=" * 80)
    print(f"OVERALL EVALUATION VERIFICATION: {'ALL TESTS PASSED [OK]' if all_passed else 'FAIL'}")
    print("=" * 80)

    with open("results/verification_report.json", "w") as f:
        json.dump({'all_passed': all_passed, 'results': results}, f, indent=2)
    print("Verification metrics saved to results/verification_report.json")

if __name__ == "__main__":
    run_all_tests()
