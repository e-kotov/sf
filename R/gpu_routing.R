#' Query or set GPU acceleration in sf
#'
#' @param use_gpu logical; if specified, enables (TRUE) or disables (FALSE) GPU acceleration.
#' @param force logical; if TRUE, forces GPU execution even for small datasets below the crossover threshold.
#' @return Logical indicating whether GPU acceleration is currently active.
#' @export
sf_use_gpu = function(use_gpu, force = FALSE) {
	ret_val = getOption("sf_use_gpu", default = FALSE)
	if (!missing(use_gpu)) {
		stopifnot(is.logical(use_gpu), length(use_gpu) == 1, !is.na(use_gpu))
		if (ret_val != use_gpu) {
			backend = getOption("sf_gpu_backend", default = "auto")
			message(paste0("GPU acceleration switched ", ifelse(use_gpu, "on", "off"), " [backend: ", backend, "]"))
		}
		options(sf_use_gpu = use_gpu)
		options(sf_gpu_force = isTRUE(force))
		invisible(ret_val)
	} else {
		ret_val
	}
}

#' Query or set active GPU backend
#'
#' @param backend character; one of "auto", "cuda", "metal", "wgpu", "cpu".
#' @export
sf_gpu_backend = function(backend) {
	cur = getOption("sf_gpu_backend", default = "auto")
	if (!missing(backend)) {
		if (identical(backend, "cuspatial")) backend = "cuda"
		valid = c("auto", "cuda", "metal", "wgpu", "cpu")
		stopifnot(backend %in% valid)
		options(sf_gpu_backend = backend)
		invisible(cur)
	} else {
		cur
	}
}

#' Query or set size-based crossover thresholds for GPU auto-routing
#'
#' @param op character; operation name, e.g. "distance", "pip", "join", "transform".
#' @param value numeric; threshold value. When data volume (e.g. total pairs or points) is below this value, sf routes to CPU.
#' @export
sf_gpu_threshold = function(op = c("distance", "pip", "join", "transform"), value) {
	op = match.arg(op)
	opt_name = paste0("sf_gpu_threshold_", op)
	default_thresholds = c(
		distance = 25000,    # 25k pairs (calibrated empirically on A100 HPC: 19x-24x speedup at 250k, break-even at 10k-20k)
		pip = 5000,          # 5,000 points against polygons
		join = 5000,         # 5,000 points in spatial join
		transform = 20000    # 20,000 coordinate pairs
	)
	cur = getOption(opt_name, default = default_thresholds[[op]])
	if (!missing(value)) {
		stopifnot(is.numeric(value), length(value) == 1, value >= 0)
		opts = list()
		opts[[opt_name]] = value
		options(opts)
		invisible(cur)
	} else {
		cur
	}
}

#' Determine if GPU execution should be used for an operation
#'
#' Evaluates whether GPU acceleration is enabled and whether dataset size justifies
#' GPU kernel launch and host-device transfer latency.
#'
#' @param op character; "distance", "pip", "join", or "transform".
#' @param size numeric; size metric (e.g. N * M pairs for distance, N points for PIP).
#' @return logical
#' @keywords internal
sf_should_use_gpu = function(op, size) {
	if (!isTRUE(getOption("sf_use_gpu", default = FALSE)))
		return(FALSE)
	if (isTRUE(getOption("sf_gpu_force", default = FALSE)))
		return(TRUE)
	thresh = sf_gpu_threshold(op)
	return(size >= thresh)
}
