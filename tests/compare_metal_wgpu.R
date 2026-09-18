source("R/wgpu.R")
source("R/metal.R")
library(sf)

# Warmup both engines
m_warm <- matrix(runif(200), 100, 2)
invisible(st_distance_wgpu(m_warm, m_warm))
invisible(st_distance_metal(m_warm, m_warm))

sizes <- c(2500, 5000, 10000, 20000)

cat(sprintf("%-7s | %-10s | %-12s | %-12s | %-12s | %-12s\n",
            "N", "Pairs", "Metal (Native)", "wgpu (Rust)", "sf (CPU)", "Metal vs wgpu"))
cat(paste(rep("-", 76), collapse = ""), "\n")

for (n in sizes) {
  m1 <- matrix(runif(2 * n, 0, 1000), ncol = 2)
  m2 <- matrix(runif(2 * n, 0, 1000), ncol = 2)

  # 1. Native Metal
  t_metal <- system.time({ res_metal <- st_distance_metal(m1, m2) })["elapsed"]

  # 2. wgpu
  t_wgpu <- system.time({ res_wgpu <- st_distance_wgpu(m1, m2) })["elapsed"]

  # 3. CPU sf
  if (n <= 5000) {
    s1 <- st_as_sf(as.data.frame(m1), coords = 1:2)
    s2 <- st_as_sf(as.data.frame(m2), coords = 1:2)
    t_cpu <- system.time({ res_cpu <- as.matrix(st_distance(s1, s2)) })["elapsed"]
    t_cpu_str <- sprintf("%.3fs", t_cpu)
  } else {
    t_cpu_str <- "skipped"
  }

  pairs_str <- sprintf("%dM", as.integer((as.double(n) * as.double(n)) / 1e6))
  speedup_str <- sprintf("%.1fx faster", t_wgpu / t_metal)

  cat(sprintf("%-7d | %-10s | %-12s | %-12s | %-12s | %-12s\n",
              n, pairs_str, sprintf("%.3fs", t_metal), sprintf("%.3fs", t_wgpu), t_cpu_str, speedup_str))
}
