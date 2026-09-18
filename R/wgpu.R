#' Pairwise distance on GPU using wgpu (Euclidean, Haversine, S2 Chord, Vincenty Ellipsoid)
#'
#' @name st_distance_wgpu
#' @param x object of class sf, sfc, sfg or numeric matrix of 2D points
#' @param y optional second object
#' @param method character; one of "euclidean", "haversine" (cuSpatial/Sedona), "s2" (Google S2/sf default), or "vincenty" (WGS84 Spheroid)
#' @param dylib_path optional path to compiled rust_wgpu_sf dynamic library
#' @return A matrix of distances in meters (or CRS units for Euclidean)
#' @export
st_distance_wgpu <- function(x, y, method = NULL, dylib_path = NULL) {
  if (is.null(dylib_path)) {
    candidates <- c(
      file.path("src", "rust", "target", "release", "librust_wgpu_sf.dylib"),
      file.path("src", "rust", "target", "debug", "librust_wgpu_sf.dylib"),
      file.path("src", "rust", "target", "release", "librust_wgpu_sf.so"),
      file.path("src", "rust", "target", "release", "rust_wgpu_sf.dll"),
      file.path("..", "src", "rust", "target", "release", "librust_wgpu_sf.dylib"),
      system.file("libs", "librust_wgpu_sf.dylib", package = "sf")
    )
    dylib_path <- candidates[file.exists(candidates)][1]
  }

  if (is.na(dylib_path) || !file.exists(dylib_path)) {
    stop("wgpu dynamic library not found. Run cargo build in src/rust first.")
  }

  if (!is.loaded("c_wgpu_compute")) {
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
      stop("Unsupported object type")
    }
  }

  # Auto-select method if NULL
  is_ll <- if (inherits(x, c("sf", "sfc"))) isTRUE(sf::st_is_longlat(x)) else FALSE
  if (is.null(method)) {
    method <- if (is_ll) "s2" else "euclidean"
  }
  method <- match.arg(tolower(method), c("euclidean", "haversine", "s2", "vincenty"))

  method_code <- switch(method,
    euclidean = 0L,
    haversine = 1L,
    s2 = 2L,
    vincenty = 3L
  )

  p1 <- extract_pts(x)
  p2 <- if (missing(y) || is.null(y)) p1 else extract_pts(y)

  n1 <- as.integer(nrow(p1))
  n2 <- as.integer(nrow(p2))

  if (n1 == 0L || n2 == 0L) {
    return(matrix(numeric(0), nrow = n1, ncol = n2))
  }

  pts1_flat <- as.single(as.vector(t(p1)))
  pts2_flat <- as.single(as.vector(t(p2)))
  out_vec <- single(length = as.double(n1) * as.double(n2))

  res <- .C(
    "c_wgpu_compute",
    method = method_code,
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
