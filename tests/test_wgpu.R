library(sf)
source("R/wgpu.R")

cat("=== 1. Correctness Verification ===\n")
set.seed(42)
n_test <- 50
pts1 <- st_sfc(lapply(1:n_test, function(i) st_point(runif(2, 0, 100))))
pts2 <- st_sfc(lapply(1:n_test, function(i) st_point(runif(2, 0, 100))))

dist_sf <- as.matrix(st_distance(pts1, pts2))
dist_wgpu <- st_distance_wgpu(pts1, pts2)

max_diff <- max(abs(dist_sf - dist_wgpu))
cat(sprintf("Maximum absolute difference vs CPU st_distance: %e\n", max_diff))
stopifnot(max_diff < 1e-4)
cat("Verification PASSED: wgpu matches st_distance exactly within float32 precision!\n\n")

cat("=== 2. Benchmark on Apple Silicon GPU ===\n")
n_bench <- 5000
cat(sprintf("Testing %d x %d points (%d,000,000 distance pairs)...\n", 
            n_bench, n_bench, (n_bench * n_bench) / 1e6))

p_b1 <- st_sfc(lapply(1:n_bench, function(i) st_point(runif(2, 0, 1000))))
p_b2 <- st_sfc(lapply(1:n_bench, function(i) st_point(runif(2, 0, 1000))))

# Warmup GPU
invisible(st_distance_wgpu(pts1, pts2))

t_gpu <- system.time({
  gpu_res <- st_distance_wgpu(p_b1, p_b2)
})
cat(sprintf("GPU (wgpu/Metal on Apple Silicon): %.3f seconds (elapsed)\n", t_gpu["elapsed"]))

t_cpu <- system.time({
  cpu_res <- as.matrix(st_distance(p_b1, p_b2))
})
cat(sprintf("CPU (sf / GEOS): %.3f seconds (elapsed)\n", t_cpu["elapsed"]))

speedup <- t_cpu["elapsed"] / t_gpu["elapsed"]
cat(sprintf("==> GPU Speedup: %.1fx faster!\n", speedup))
