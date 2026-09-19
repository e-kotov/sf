#!/bin/bash
#SBATCH -p gpu
#SBATCH -G a100:1
#SBATCH -t 01:00:00
#SBATCH --mem=64G
#SBATCH -J sf_cuspatial_benchmark
#SBATCH -o sf_cuspatial_bench_%j.out
#SBATCH -e sf_cuspatial_bench_%j.err

echo "=== Running sf cuSpatial GPU Benchmark on GWDG HPC ==="
date
hostname
nvidia-smi

# Load required modules on GWDG
module purge
module load cuda/12.4
module load gcc/11.4
module load R/4.4.0

# Verify R & CUDA
echo "CUDA Version:"
nvcc --version
echo "R Version:"
R --version | head -n 1

# Execute benchmark harness
Rscript inst/benchmarks/gwdg_hpc_benchmark.R

echo "=== Benchmark Complete ==="
date
