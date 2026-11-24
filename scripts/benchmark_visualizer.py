import os
import json
import numpy as np

# Set writable cache directory for Matplotlib on Windows
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_charts_from_data():
    os.makedirs("results/plots", exist_ok=True)
    with open("results/benchmark_data.json", "r") as f:
        benchmarks = json.load(f)

    # Filter for Softmax and LayerNorm
    sm_data = [b for b in benchmarks if b["operation"] == "Softmax"]
    ln_data = [b for b in benchmarks if b["operation"] == "LayerNorm"]

    # 1. Softmax Latency Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    labels = [b["variant"].split(". ")[1] for b in sm_data]
    times = [b["time_ms"] for b in sm_data]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    bars = ax.bar(labels, times, color=colors, edgecolor='black', width=0.55)
    ax.set_ylabel('Execution Time (ms) - Lower is Better', fontsize=11, fontweight='bold')
    ax.set_title('Softmax Latency on NVIDIA Tesla T4 (4096 x 1024)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, b in zip(bars, sm_data):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f} ms\n({b['speedup']:.2f}x)", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=15, ha='right', fontsize=10, fontweight='semibold')
    plt.tight_layout()
    plt.savefig("results/plots/softmax_latency_comparison.png")
    plt.close()

    # 2. LayerNorm Latency Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    labels_ln = [b["variant"].split(". ")[1] for b in ln_data]
    times_ln = [b["time_ms"] for b in ln_data]

    bars = ax.bar(labels_ln, times_ln, color=colors, edgecolor='black', width=0.55)
    ax.set_ylabel('Execution Time (ms) - Lower is Better', fontsize=11, fontweight='bold')
    ax.set_title('LayerNorm Latency on NVIDIA Tesla T4 (4096 x 1024)', fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, b in zip(bars, ln_data):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f} ms\n({b['speedup']:.2f}x)", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=15, ha='right', fontsize=10, fontweight='semibold')
    plt.tight_layout()
    plt.savefig("results/plots/layernorm_latency_comparison.png")
    plt.close()

    # 3. Effective Memory Bandwidth Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    all_labels = [f"SM: {b['variant'].split('. ')[1]}" for b in sm_data] + [f"LN: {b['variant'].split('. ')[1]}" for b in ln_data]
    all_bws = [b["bandwidth_gb_s"] for b in sm_data] + [b["bandwidth_gb_s"] for b in ln_data]
    bar_colors = ['#3498db']*4 + ['#2ecc71']*4

    bars = ax.barh(all_labels, all_bws, color=bar_colors, edgecolor='black', height=0.6)
    ax.set_xlabel('Effective Bandwidth (GB/s) - Higher is Better', fontsize=11, fontweight='bold')
    ax.set_title('Effective Memory Bandwidth on NVIDIA Tesla T4 GPU (Peak ~320 GB/s GDDR6)', fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 2.0, bar.get_y() + bar.get_height()/2.0, f"{w:.1f} GB/s", 
                ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("results/plots/effective_bandwidth_comparison.png")
    plt.close()

    # 4. Speedup Factor Summary Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    sm_labels_short = ["Shared Mem", "1 Warp/Row", "Online (1-Pass)", "Vectorized"]
    sm_speeds = [b["speedup"] for b in sm_data]
    bars1 = ax1.bar(sm_labels_short, sm_speeds, color=['#95a5a6', '#3498db', '#2ecc71', '#9b59b6'], edgecolor='black', width=0.55)
    ax1.set_ylabel('Speedup Multiplier vs. Baseline', fontsize=11, fontweight='bold')
    ax1.set_title('Softmax Speedup (NVIDIA T4)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, y + 0.08, f"{y:.2f}x", ha='center', va='bottom', fontweight='bold')

    ln_labels_short = ["Shared Mem", "1 Warp/Row", "Welford", "Vectorized"]
    ln_speeds = [b["speedup"] for b in ln_data]
    bars2 = ax2.bar(ln_labels_short, ln_speeds, color=['#95a5a6', '#3498db', '#2ecc71', '#9b59b6'], edgecolor='black', width=0.55)
    ax2.set_ylabel('Speedup Multiplier vs. Baseline', fontsize=11, fontweight='bold')
    ax2.set_title('LayerNorm Speedup (NVIDIA T4)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars2:
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, y + 0.05, f"{y:.2f}x", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig("results/plots/warp_speedup_scaling.png")
    plt.close()
    print("All plots successfully updated with NVIDIA Tesla T4 benchmark numbers!")

if __name__ == "__main__":
    plot_charts_from_data()
