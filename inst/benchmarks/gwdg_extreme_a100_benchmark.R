# GWDG HPC Extreme-Scale Benchmarking Harness for NVIDIA A100-80GB vs CPU sf
# Tests Multi-Billion Pair Matrices (up to 10 Billion pairs) and Extreme Scaling Curves

suppressPackageStartupMessages({
	library(sf)
})

# Locate sf source root
args = commandArgs(trailingOnly = FALSE)
file_arg = grep("^--file=", args, value = TRUE)
script_path = if (length(file_arg)) sub("^--file=", "", file_arg) else "inst/benchmarks/gwdg_extreme_a100_benchmark.R"
sf_root = normalizePath(file.path(dirname(script_path), "..", ".."))

# Source GPU modules
source(file.path(sf_root, "R", "gpu_detect.R"))
source(file.path(sf_root, "R", "gpu_routing.R"))
source(file.path(sf_root, "R", "cuspatial.R"))
source(file.path(sf_root, "R", "geom-measures.R"))
environment(st_distance) <- asNamespace("sf")
assignInNamespace("st_distance", st_distance, ns = "sf")

# Load compiled bridge
bridge_so = file.path(sf_root, "src", "cuspatial_bridge.so")
if (file.exists(bridge_so)) {
	tryCatch(dyn.load(bridge_so), error = function(e) cat("Error loading bridge:", e$message, "\n"))
}
sf_gpu_backend("cuspatial")

cat("================================================================\n")
cat("      sf Extreme-Scale GPU Benchmark (NVIDIA A100-80GB)         \n")
cat("================================================================\n\n")

gpu_info = sf_detect_gpu()
cat(sprintf("Active GPU Backend: %s\n", gpu_info$backend))
cat(sprintf("Device Name:        %s\n\n", gpu_info$details$device_name))

results = data.frame(
	Benchmark = character(),
	Nx = numeric(),
	Ny = numeric(),
	Total_Pairs = numeric(),
	Memory_GB = numeric(),
	Time_sec = numeric(),
	Throughput_GigaPairs_per_sec = numeric(),
	Mode = character(),
	stringsAsFactors = FALSE
)

# -------------------------------------------------------------------
# Test 1: The Billion-to-Multi-Billion Pair Scaling Frontier
# -------------------------------------------------------------------
cat("--- Part 1: The Multi-Billion Pair Frontier (Dense Distance Matrix) ---\n")

billion_scales = list(
	list(nx = 10000, ny = 10000, label = "100 Million Pairs"),
	list(nx = 25000, ny = 25000, label = "625 Million Pairs"),
	list(nx = 50000, ny = 50000, label = "2.50 Billion Pairs"),
	list(nx = 75000, ny = 75000, label = "5.62 Billion Pairs")
)

sf_use_gpu(TRUE, force = TRUE)

for (s in billion_scales) {
	nx = s$nx
	ny = s$ny
	total_pairs = as.numeric(nx) * as.numeric(ny)
	mem_gb = (total_pairs * 4) / (1024^3) # FP32 size

	cat(sprintf("\n[ALLOCATING] %s (Nx=%d, Ny=%d, Size=%.2f GB VRAM)...\n", s$label, nx, ny, mem_gb))
	
	coords_x = cbind(runif(nx, -180, 180), runif(nx, -70, 70))
	coords_y = cbind(runif(ny, -180, 180), runif(ny, -70, 70))

	# 1. Geodetic (Haversine)
	cat("  Computing Geodetic Distance on A100...")
	gc()
	t_geod = system.time({
		res = .Call("c_cuspatial_distance", coords_x, coords_y, TRUE)
	})[["elapsed"]]
	throughput_geod = (total_pairs / 1e9) / max(t_geod, 0.0001)
	cat(sprintf(" Done in %7.3fs | Throughput: %6.2f GigaPairs/s\n", t_geod, throughput_geod))

	results = rbind(results, data.frame(
		Benchmark = "dense_matrix_geodetic",
		Nx = nx,
		Ny = ny,
		Total_Pairs = total_pairs,
		Memory_GB = mem_gb,
		Time_sec = t_geod,
		Throughput_GigaPairs_per_sec = throughput_geod,
		Mode = "A100 GPU",
		stringsAsFactors = FALSE
	))
	rm(res)
	gc()

	# 2. Planar Euclidean
	cat("  Computing Euclidean Distance on A100...")
	t_eucl = system.time({
		res = .Call("c_cuspatial_distance", coords_x, coords_y, FALSE)
	})[["elapsed"]]
	throughput_eucl = (total_pairs / 1e9) / max(t_eucl, 0.0001)
	cat(sprintf(" Done in %7.3fs | Throughput: %6.2f GigaPairs/s\n", t_eucl, throughput_eucl))

	results = rbind(results, data.frame(
		Benchmark = "dense_matrix_euclidean",
		Nx = nx,
		Ny = ny,
		Total_Pairs = total_pairs,
		Memory_GB = mem_gb,
		Time_sec = t_eucl,
		Throughput_GigaPairs_per_sec = throughput_eucl,
		Mode = "A100 GPU",
		stringsAsFactors = FALSE
	))
	rm(res, coords_x, coords_y)
	gc()
}

# -------------------------------------------------------------------
# Test 2: Extreme Asymmetric 1-to-N Nearest Scanning
# -------------------------------------------------------------------
cat("\n--- Part 2: Extreme Asymmetric Vectorization (1-to-N Facility Scanning) ---\n")

asymm_scales = list(
	list(nx = 1,     ny = 10000000, label = "1 Facility x 10 Million Targets"),
	list(nx = 10,    ny = 5000000,  label = "10 Facilities x 5 Million Targets"),
	list(nx = 100,   ny = 1000000,  label = "100 Facilities x 1 Million Targets")
)

for (s in asymm_scales) {
	nx = s$nx
	ny = s$ny
	total_pairs = as.numeric(nx) * as.numeric(ny)
	mem_gb = (total_pairs * 4) / (1024^3)

	cat(sprintf("\n[TESTING] %s (%10.0f pairs)...\n", s$label, total_pairs))
	coords_x = cbind(runif(nx, -180, 180), runif(nx, -70, 70))
	coords_y = cbind(runif(ny, -180, 180), runif(ny, -70, 70))

	t_gpu = system.time({
		res = .Call("c_cuspatial_distance", coords_x, coords_y, TRUE)
	})[["elapsed"]]
	throughput = (total_pairs / 1e9) / max(t_gpu, 0.0001)
	cat(sprintf("  A100 Time: %7.4fs | Throughput: %6.2f GigaPairs/s\n", t_gpu, throughput))

	results = rbind(results, data.frame(
		Benchmark = "asymmetric_scan",
		Nx = nx,
		Ny = ny,
		Total_Pairs = total_pairs,
		Memory_GB = mem_gb,
		Time_sec = t_gpu,
		Throughput_GigaPairs_per_sec = throughput,
		Mode = "A100 GPU",
		stringsAsFactors = FALSE
	))
	rm(res, coords_x, coords_y)
	gc()
}

# -------------------------------------------------------------------
# Summary & Export
# -------------------------------------------------------------------
cat("\n================================================================\n")
cat("                Extreme Scaling Benchmark Summary\n")
cat("================================================================\n")
print(results[, c("Benchmark", "Nx", "Ny", "Total_Pairs", "Memory_GB", "Time_sec", "Throughput_GigaPairs_per_sec")])

dir.create("inst/benchmarks", showWarnings = FALSE, recursive = TRUE)
write.csv(results, "inst/benchmarks/extreme_a100_scaling.csv", row.names = FALSE)
cat("\nResults saved to inst/benchmarks/extreme_a100_scaling.csv\n")
