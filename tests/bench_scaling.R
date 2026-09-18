source("R/wgpu.R")
library(sf)

sizes <- c(1000, 2500, 5000, 7500, 10000)

cat(sprintf("%-8s | %-10s | %-12s | %-12s | %-8s\n", 
            "N", "Pairs", "wgpu (GPU)", "sf (CPU)", "Speedup"))
cat(paste(rep("-", 58), collapse = ""), "\n")

for (n in sizes) {
  m1 <- matrix(runif(2 * n, 0, 1000), ncol = 2)
  m2 <- matrix(runif(2 * n, 0, 1000), ncol = 2)
  
  # GPU run
  t_gpu <- system.time({
    res_gpu <- st_distance_wgpu(m1, m2)
  })["elapsed"]
  
  # CPU run using st_distance
  s1 <- st_as_sf(as.data.frame(m1), coords = 1:2)
  s2 <- st_as_sf(as.data.frame(m2), coords = 1:2)
  
  t_cpu <- system.time({
    res_cpu <- as.matrix(st_distance(s1, s2))
  })["elapsed"]
  
  speedup <- sprintf("%.2fx", t_cpu / t_gpu)
  pairs_str <- sprintf("%dM", as.integer((n * n) / 1e6))
  cat(sprintf("%-8d | %-10s | %-12s | %-12s | %-8s\n", 
              n, pairs_str, sprintf("%.3fs", t_gpu), sprintf("%.3fs", t_cpu), speedup))
}
