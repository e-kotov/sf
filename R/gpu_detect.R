#' Detect available GPU acceleration backends
#'
#' Probes the runtime environment for NVIDIA CUDA (cuSpatial), Apple Silicon Metal,
#' and cross-platform WebGPU (wgpu).
#'
#' @return A list containing `has_gpu` (logical), `backend` (character), and `details` (list).
#' @export
sf_detect_gpu = function() {
	has_gpu = FALSE
	backend = "cpu"
	details = list(metal = FALSE, cuda = FALSE, wgpu = FALSE, device_name = "None")

	# 1. Probe Apple Silicon Metal (macOS)
	if (Sys.info()[["sysname"]] == "Darwin") {
		# Check if Metal framework is available and dylib exists
		metal_lib = system.file("libs", "libmetal_sf.dylib", package = "sf")
		if (file.exists(metal_lib) || file.exists("src/libmetal_sf.dylib")) {
			details$metal = TRUE
			has_gpu = TRUE
			backend = "metal"
			details$device_name = "Apple Silicon GPU (Metal)"
		}
	}

	# 2. Probe NVIDIA CUDA (cuSpatial)
	cuda_count = 0
	tryCatch({
		cuda_count = .Call("c_cuda_device_count", PACKAGE = "sf")
	}, error = function(e) {
		# Fallback if symbol not yet registered
	})
	if (cuda_count > 0) {
		details$cuda = TRUE
		has_gpu = TRUE
		backend = "cuspatial"
		details$device_name = paste0("NVIDIA CUDA GPU (", cuda_count, " device", if (cuda_count > 1) "s" else "", ")")
	}

	# 3. Probe WebGPU (Rust wgpu)
	wgpu_avail = FALSE
	tryCatch({
		wgpu_avail = c_wgpu_is_available()
	}, error = function(e) {
		# Fallback
	})
	if (wgpu_avail) {
		details$wgpu = TRUE
		if (!has_gpu) {
			has_gpu = TRUE
			backend = "wgpu"
			details$device_name = "WebGPU (wgpu-native)"
		}
	}

	list(has_gpu = has_gpu, backend = backend, details = details)
}
