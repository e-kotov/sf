source("R/wgpu.R")
library(sf)

cat("=========================================================================\n")
cat("      GEODETIC DISTANCE BENCHMARK & COMPARISON (100M PAIRWISE EVALS)     \n")
cat("=========================================================================\n\n")

# 1. Verification on Global Cities
cities <- st_sf(
  name = c("London", "Paris", "New York", "Tokyo"),
  geom = st_sfc(
    st_point(c(-0.1276, 51.5074)),
    st_point(c(2.3522, 48.8566)),
    st_point(c(-74.0060, 40.7128)),
    st_point(c(139.6917, 35.6895)),
    crs = 4326
  )
)

d_sf_s2   <- as.matrix(units::drop_units(st_distance(cities)))
d_karney  <- as.matrix(units::drop_units(lwgeom::st_geod_distance(cities, cities)))
d_gpu_hav <- st_distance_wgpu(cities, method = "haversine")
d_gpu_s2  <- st_distance_wgpu(cities, method = "s2")
d_gpu_vic <- st_distance_wgpu(cities, method = "vincenty")

cat("--- London to Paris (343 km) ---\n")
cat(sprintf("  sf default (S2 Sphere CPU):      %.2f m\n", d_sf_s2[1,2]))
cat(sprintf("  GPU S2 Chord (Google S2):        %.2f m (diff: %.2f m)\n", d_gpu_s2[1,2], abs(d_gpu_s2[1,2] - d_sf_s2[1,2])))
cat(sprintf("  GPU Haversine (cuSpatial/Sedona):%.2f m (diff: %.2f m)\n", d_gpu_hav[1,2], abs(d_gpu_hav[1,2] - d_sf_s2[1,2])))
cat(sprintf("  lwgeom Karney (WGS84 CPU):       %.2f m\n", d_karney[1,2]))
cat(sprintf("  GPU Vincenty (WGS84 Ellipsoid):  %.2f m (diff vs Karney: %.2f m)\n\n", d_gpu_vic[1,2], abs(d_gpu_vic[1,2] - d_karney[1,2])))

cat("--- New York to Tokyo (10,848 km) ---\n")
cat(sprintf("  sf default (S2 Sphere CPU):      %.2f m\n", d_sf_s2[3,4]))
cat(sprintf("  GPU S2 Chord (Google S2):        %.2f m (diff: %.2f m)\n", d_gpu_s2[3,4], abs(d_gpu_s2[3,4] - d_sf_s2[3,4])))
cat(sprintf("  GPU Haversine (cuSpatial/Sedona):%.2f m (diff: %.2f m)\n", d_gpu_hav[3,4], abs(d_gpu_hav[3,4] - d_sf_s2[3,4])))
cat(sprintf("  lwgeom Karney (WGS84 CPU):       %.2f m\n", d_karney[3,4]))
cat(sprintf("  GPU Vincenty (WGS84 Ellipsoid):  %.2f m (diff vs Karney: %.2f m)\n\n", d_gpu_vic[3,4], abs(d_gpu_vic[3,4] - d_karney[3,4])))

# 2. Performance Benchmark on 10,000 Global Points (100,000,000 pairs)
n <- 10000
m <- cbind(runif(n, -180, 180), runif(n, -80, 80))

cat(sprintf("--- Performance on %d Global Points (100M Pairs) ---\n", n))
t_hav <- system.time(st_distance_wgpu(m, method = "haversine"))["elapsed"]
cat(sprintf("  1. GPU Haversine (cuSpatial/Sedona): %.3f s\n", t_hav))

t_s2 <- system.time(st_distance_wgpu(m, method = "s2"))["elapsed"]
cat(sprintf("  2. GPU S2 Chord (Google S2):         %.3f s\n", t_s2))

t_vic <- system.time(st_distance_wgpu(m, method = "vincenty"))["elapsed"]
cat(sprintf("  3. GPU Vincenty (WGS84 Ellipsoid):   %.3f s\n", t_vic))
