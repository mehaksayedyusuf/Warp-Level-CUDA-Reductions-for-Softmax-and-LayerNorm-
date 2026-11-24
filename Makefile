# Makefile for PGC Evaluation: Warp-Level CUDA Reductions for Softmax and LayerNorm

NVCC := nvcc
NVCC_FLAGS := -O3 -use_fast_math -std=c++17 -arch=sm_80 --ptxas-options=-v
INCLUDES := -I./src -I./cpu_reference

PYTHON := python

SRC_CUDA := src/benchmark_main.cu
TARGET_CUDA := benchmark_warp.exe

all: $(TARGET_CUDA) verify plots

$(TARGET_CUDA): $(SRC_CUDA)
	@echo "Compiling CUDA Benchmark Suite with NVCC..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SRC_CUDA) -o $(TARGET_CUDA)

verify:
	@echo "Running Algorithmic & Numerical Verification..."
	$(PYTHON) cpu_reference/verify_numerics.py

plots:
	@echo "Generating Performance Plots & Visualizations..."
	$(PYTHON) scripts/benchmark_visualizer.py

clean:
	rm -f $(TARGET_CUDA) *.exp *.lib
	rm -rf results/plots/*.png results/*.json

.PHONY: all verify plots clean
