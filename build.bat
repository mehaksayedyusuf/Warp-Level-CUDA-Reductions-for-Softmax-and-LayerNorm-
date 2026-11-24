@echo off
echo =========================================================================
echo  PGC EVALUATION: Building Warp-Level CUDA Reductions Experiment
echo =========================================================================

where nvcc >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] NVCC detected. Compiling CUDA benchmark suite...
    nvcc -O3 -use_fast_math -std=c++17 -arch=sm_80 src\benchmark_main.cu -o benchmark_warp.exe
    if %ERRORLEVEL% EQU 0 (
        echo [SUCCESS] benchmark_warp.exe built successfully!
    ) else (
        echo [ERROR] CUDA build failed.
    )
) else (
    echo [WARNING] nvcc was not found in PATH. Skipping binary compilation.
    echo [INFO] You can compile this on any system with NVIDIA CUDA Toolkit installed.
)

echo.
echo [INFO] Running Python numerical verification suite...
python cpu_reference\verify_numerics.py

echo.
echo [INFO] Generating benchmark visualization plots...
python scripts\benchmark_visualizer.py

echo =========================================================================
echo  Build and Evaluation Pipeline Complete!
echo =========================================================================
