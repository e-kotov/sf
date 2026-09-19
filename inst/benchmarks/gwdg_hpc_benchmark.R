# GWDG HPC Benchmarking Harness for NVIDIA cuSpatial vs CPU sf
# Tests across 2D asymmetric dataset sizes (Nx x Ny) to determine exact crossover thresholds (N*)

suppressPackageStartupMessages({
	library(sf)
})

# Locate sf source root
args = commandArgs(trailingOnly = FALSE)
file_arg = grep("^--file=", args, value = TRUE)
script_path = if (length(file_arg)) sub("^--file=", "", file_arg) else "inst/benchmarks/gwdg_hpc_benchmark.R"
sf_root = normalizePath(file.path(dirname(script_path), "..", ".."))

# Source new GPU acceleration modules
source(file.path(sf_root, "R", "gpu_detect.R"))
source(file.path(sf_root, "R", "gpu_routing.R"))
source(file.path(sf_root, "R", "cuspatial.R"))
source(file.path(sf_root, "R", "geom-measures.R"))

# Compile and load bridge if not already loaded
bridge_so = file.path(sf_root, "src", "cuspatial_bridge.so")
if (!file.exists(bridge_so)) {
	system(paste("cd", file.path(sf_root, "src"), "&& R CMD SHLIB cuspatial_bridge.cpp"))
}
if (file.exists(bridge_so)) {
	tryCatch(dyn.load(bridge_so), error = function(e) NULL)
}

cat("================================================================\n")
cat("       sf GPU Acceleration Benchmarking Matrix (GWDG HPC)       \n")
cat("================================================================\n\n")

gpu_info = sf_detect_gpu()
cat(sprintf("Active GPU Backend: %s\n", gpu_info$backend))
cat(sprintf("Device Name:        %s\n\n", gpu_info$details$device_name))

results = data.frame(
	Operation = character(),
	Nx = integer(),
	Ny = integer(),
	Category = character(),
	Total_Pairs = numeric(),
	CPU_Time_sec = numeric(),
	GPU_Time_sec = numeric(),
	Speedup = numeric(),
	stringsAsFactors = FALSE
)

# 2D Asymmetric Grid: Small/Large combinations
asymmetric_grid = list(
	list(nx = 50,    ny = 50,    cat = "Small x Small"),
	list(nx = 100,   ny = 100,   cat = "Small x Small"),
	list(nx = 10,    ny = 1000,  cat = "Small x Medium"),
	list(nx = 1000,  ny = 10,    cat = "Medium x Small"),
	list(nx = 10,    ny = 25000, cat = "Small x Large"),
	list(nx = 25000, ny = 10,    cat = "Large x Small"),
	list(nx = 1000,  ny = 1000,  cat = "Medium x Medium"),
	list(nx = 5000,  ny = 5000,  cat = "Large x Large"),
	list(nx = 10000, ny = 10000, cat = "Ultra x Ultra")
)

# Benchmark 1: Planar Euclidean Distance (2D Grid)
cat("--- Running Benchmark 1: Planar Euclidean Distance (Asymmetric 2D Grid) ---\n")
for (g in asymmetric_grid) {
	nx = g$nx
	ny = g$ny
	cat_label = g$cat
	total_pairs = as.numeric(nx) * as.numeric(ny)

	pts_x = st_sfc(lapply(1:nx, function(i) st_point(c(runif(1, 0, 1000), runif(1, 0, 1000)))))
	pts_y = st_sfc(lapply(1:ny, function(i) st_point(c(runif(1, 0, 1000), runif(1, 0, 1000)))))

	# Force CPU
	sf_use_gpu(FALSE)
	t_cpu = system.time({
		d_cpu = st_distance(pts_x, pts_y)
	})[["elapsed"]]

	# Force GPU
	sf_use_gpu(TRUE, force = TRUE)
	t_gpu = system.time({
		d_gpu = st_distance(pts_x, pts_y)
	})[["elapsed"]]

	speedup = t_cpu / max(t_gpu, 0.0001)
	cat(sprintf("[%15s] Nx=%5d, Ny=%5d (%10.0f pairs): CPU = %7.3fs | GPU = %7.3fs | Speedup = %6.1fx\n",
		cat_label, nx, ny, total_pairs, t_cpu, t_gpu, speedup))

	results = rbind(results, data.frame(
		Operation = "distance_euclidean",
		Nx = nx,
		Ny = ny,
		Category = cat_label,
		Total_Pairs = total_pairs,
		CPU_Time_sec = t_cpu,
		GPU_Time_sec = t_gpu,
		Speedup = speedup
	))
}

# Benchmark 2: Spherical Geodetic Distance (Asymmetric 2D Grid)
cat("\n--- Running Benchmark 2: Spherical Geodetic Distance (Asymmetric 2D Grid) ---\n")
for (g in asymmetric_grid) {
	nx = g$nx
	ny = g$ny
	cat_label = g$cat
	total_pairs = as.numeric(nx) * as.numeric(ny)

	pts_x = st_sfc(lapply(1:nx, function(i) st_point(c(runif(1, -180, 180), runif(1, -70, 70)))), crs = "OGC:CRS84")
	pts_y = st_sfc(lapply(1:ny, function(i) st_point(c(runif(1, -180, 180), runif(1, -70, 70)))), crs = "OGC:CRS84")

	# Force CPU (s2)
	sf_use_gpu(FALSE)
	t_cpu = system.time({
		d_cpu = st_distance(pts_x, pts_y)
	})[["elapsed"]]

	# Force GPU (cuSpatial or wgpu)
	sf_use_gpu(TRUE, force = TRUE)
	t_gpu = system.time({
		d_gpu = st_distance(pts_x, pts_y)
	})[["elapsed"]]

	speedup = t_cpu / max(t_gpu, 0.0001)
	cat(sprintf("[%15s] Nx=%5d, Ny=%5d (%10.0f pairs): CPU = %7.3fs | GPU = %7.3fs | Speedup = %6.1fx\n",
		cat_label, nx, ny, total_pairs, t_cpu, t_gpu, speedup))

	results = rbind(results, data.frame(
		Operation = "distance_geodetic",
		Nx = nx,
		Ny = ny,
		Category = cat_label,
		Total_Pairs = total_pairs,
		CPU_Time_sec = t_cpu,
		GPU_Time_sec = t_gpu,
		Speedup = speedup
	))
}

# Output Summary & Crossover Analysis
cat("\n================================================================\n")
cat("            2D Asymmetric Crossover Summary\n")
cat("================================================================\n")

for (op in unique(results$Operation)) {
	cat(sprintf("\nOperation: %s\n", op))
	sub = results[results$Operation == op, ]
	for (i in 1:nrow(sub)) {
		status = if (sub$GPU_Time_sec[i] < sub$CPU_Time_sec[i]) "GPU WINS" else "CPU WINS"
		cat(sprintf("  [%s] Nx=%5d x Ny=%5d (%10.0f pairs): %s (%.1fx)\n",
			sub$Category[i], sub$Nx[i], sub$Ny[i], sub$Total_Pairs[i], status, sub$Speedup[i]))
	}
}

dir.create("inst/benchmarks", showWarnings = FALSE, recursive = TRUE)
write.csv(results, "inst/benchmarks/crossover_thresholds.csv", row.names = FALSE)
cat("\nResults saved to inst/benchmarks/crossover_thresholds.csv\n")
