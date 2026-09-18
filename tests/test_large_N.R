source("R/wgpu.R")
source("R/metal.R")

sizes <- c(25000, 30000)

cat(sprintf("%-7s | %-12s | %-12s | %-14s | %-14s | %-12s\n",
            "N", "Pairs", "RAM Needed", "Metal (Native)", "wgpu (Rust)", "Metal Speedup"))
cat(paste(rep("-", 85), collapse = ""), "\n")

for (n in sizes) {
  pairs <- as.double(n) * as.double(n)
  ram_gb <- (pairs * 4) / (1024^3)
  
  cat(sprintf("Allocating inputs for N = %d (%.2f GB matrix)...\n", n, ram_gb))
  m1 <- matrix(runif(2 * n, 0, 1000), ncol = 2)
  m2 <- matrix(runif(2 * n, 0, 1000), ncol = 2)

  # 1. Native Metal
  cat("Running Native Metal...\n")
  t_metal <- tryCatch({
    system.time({ res_metal <- st_distance_metal(m1, m2) })["elapsed"]
  }, error = function(e) {
    cat(sprintf("Metal error: %s\n", e$message))
    NA
  })

  # 2. wgpu
  cat("Running wgpu...\n")
  t_wgpu <- tryCatch({
    system.time({ res_wgpu <- st_distance_wgpu(m1, m2) })["elapsed"]
  }, error = function(e) {
    cat(sprintf("wgpu error: %s\n", e$message))
    NA
  })

  metal_str <- if (is.na(t_metal)) "FAILED" else sprintf("%.2fs", t_metal)
  wgpu_str  <- if (is.na(t_wgpu)) "FAILED" else sprintf("%.2fs", t_wgpu)
  ratio_str <- if (!is.na(t_metal) && !is.na(t_wgpu)) sprintf("%.2fx faster", t_wgpu / t_metal) else "N/A"
  
  pairs_str <- sprintf("%dM", as.integer(pairs / 1e6))
  cat(sprintf("%-7d | %-12s | %-12s | %-14s | %-14s | %-12s\n",
              n, pairs_str, sprintf("%.2f GB", ram_gb), metal_str, wgpu_str, ratio_str))
  
  # Clean up memory
  rm(res_metal, res_wgpu, m1, m2)
  gc()
}
