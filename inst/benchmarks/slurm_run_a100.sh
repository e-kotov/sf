#!/bin/bash
#SBATCH --account=scc_mrdf_all
#SBATCH --partition=scc-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=sf_cuda_benchmark
#SBATCH --output=.ws_storage/logs/slurm/%x-%j.out
#SBATCH --wckey=sf-gpu

echo "=== Running sf CUDA GPU Benchmark on GWDG HPC ==="
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
