#' Pairwise Euclidean distance matrix on Apple Silicon GPU using native Metal
#'
#' @name st_distance_metal
#' @param x object of class sf, sfc, sfg or numeric matrix of 2D points
#' @param y optional second object
#' @param dylib_path path to libmetal_sf.dylib
#' @return A matrix of Euclidean distances
#' @export
st_distance_metal <- function(x, y, dylib_path = NULL) {
  if (is.null(dylib_path)) {
    candidates <- c(
      file.path("src", "libmetal_sf.dylib"),
      file.path("..", "src", "libmetal_sf.dylib"),
      system.file("libs", "libmetal_sf.dylib", package = "sf")
    )
    dylib_path <- candidates[file.exists(candidates)][1]
  }

  if (is.na(dylib_path) || !file.exists(dylib_path)) {
    stop("libmetal_sf.dylib not found. Compile src/metal_dist.mm first.")
  }

  if (!is.loaded("c_metal_distance_matrix")) {
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
    "c_metal_distance_matrix",
    n1 = n1,
    pts1 = pts1_flat,
    n2 = n2,
    pts2 = pts2_flat,
    out = out_vec
  )

  matrix(res$out, nrow = n1, ncol = n2)
}
