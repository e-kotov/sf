#' Pairwise Euclidean distance matrix on GPU using wgpu (WebGPU/Metal/Vulkan)
#'
#' @name st_distance_wgpu
#' @param x object of class \code{sf}, \code{sfc}, \code{sfg} or matrix/data.frame of 2D points
#' @param y optional second object; if missing, pairwise distances within \code{x} are computed
#' @param dylib_path optional path to the compiled rust_wgpu_sf dynamic library
#' @return A matrix of Euclidean distances (in CRS units) with dimensions length(x) by length(y)
#' @export
st_distance_wgpu <- function(x, y, dylib_path = NULL) {
  if (is.null(dylib_path)) {
    candidates <- c(
      file.path("src", "rust", "target", "debug", "librust_wgpu_sf.dylib"),
      file.path("src", "rust", "target", "release", "librust_wgpu_sf.dylib"),
      file.path("src", "rust", "target", "debug", "librust_wgpu_sf.so"),
      file.path("src", "rust", "target", "release", "librust_wgpu_sf.so"),
      file.path("src", "rust", "target", "debug", "rust_wgpu_sf.dll"),
      file.path("src", "rust", "target", "release", "rust_wgpu_sf.dll"),
      file.path("..", "src", "rust", "target", "debug", "librust_wgpu_sf.dylib"),
      system.file("libs", "librust_wgpu_sf.dylib", package = "sf")
    )
    dylib_path <- candidates[file.exists(candidates)][1]
  }

  if (is.na(dylib_path) || !file.exists(dylib_path)) {
    stop("wgpu dynamic library not found. Run cargo build in src/rust first.")
  }

  if (!is.loaded("c_wgpu_distance_matrix")) {
    dyn.load(dylib_path)
  }

  extract_pts <- function(obj) {
    if (inherits(obj, c("sf", "sfc", "sfg"))) {
      crds <- sf::st_coordinates(obj)
      if (ncol(crds) < 2) stop("Geometries must have at least 2 dimensions")
      crds[, 1:2, drop = FALSE]
    } else if (is.matrix(obj) || is.data.frame(obj)) {
      as.matrix(obj)[, 1:2, drop = FALSE]
    } else {
      stop("Unsupported object type for st_distance_wgpu")
    }
  }

  p1 <- extract_pts(x)
  if (missing(y) || is.null(y)) {
    p2 <- p1
  } else {
    p2 <- extract_pts(y)
  }

  n1 <- as.integer(nrow(p1))
  n2 <- as.integer(nrow(p2))

  if (n1 == 0L || n2 == 0L) {
    return(matrix(numeric(0), nrow = n1, ncol = n2))
  }

  # Interleaved coordinates [x0, y0, x1, y1, ...] as single-precision float
  pts1_flat <- as.single(as.vector(t(p1)))
  pts2_flat <- as.single(as.vector(t(p2)))

  out_vec <- single(length = as.double(n1) * as.double(n2))

  res <- .C(
    "c_wgpu_distance_matrix",
    n1 = n1,
    pts1 = pts1_flat,
    n2 = n2,
    pts2 = pts2_flat,
    out = out_vec
  )

  out_mat <- matrix(res$out, nrow = n1, ncol = n2)
  if (!is.null(row.names(x))) rownames(out_mat) <- row.names(x)
  if (!missing(y) && !is.null(row.names(y))) colnames(out_mat) <- row.names(y)

  out_mat
}
