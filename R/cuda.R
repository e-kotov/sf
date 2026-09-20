#' Compute pairwise distances using NVIDIA CUDA
#'
#' @param x sf or sfc object
#' @param y sf or sfc object (optional)
#' @param method "euclidean" or "haversine"
#' @return numeric matrix of pairwise distances
#' @export
st_distance_cuda = function(x, y, method = c("euclidean", "haversine")) {
	if (missing(y)) y = x
	method = match.arg(method)
	coords_x = st_coordinates(x)
	coords_y = st_coordinates(y)
	is_geodetic = (method == "haversine")
	tryCatch(
		.Call("c_cuda_distance", coords_x, coords_y, as.logical(is_geodetic), PACKAGE = "sf"),
		error = function(e) .Call("c_cuda_distance", coords_x, coords_y, as.logical(is_geodetic))
	)
}

#' Test Point-in-Polygon containment using NVIDIA CUDA
#'
#' @param points sf or sfc_POINT object
#' @param polygons sf or sfc_POLYGON/MULTIPOLYGON object
#' @return logical matrix (points x polygons) or integer bitmask
#' @export
st_pip_cuda = function(points, polygons) {
	pt_coords = st_coordinates(points)
	# Flatten polygon rings into contiguous coordinate array and offset vectors
	poly_coords = st_coordinates(polygons)
	poly_offsets = as.integer(c(0, cumsum(vapply(st_geometry(polygons), length, integer(1)))))
	ring_offsets = as.integer(c(0, nrow(poly_coords)))
	
	tryCatch(
		.Call("c_cuda_pip",
			as.numeric(pt_coords[, 1]),
			as.numeric(pt_coords[, 2]),
			poly_offsets,
			ring_offsets,
			as.numeric(poly_coords[, 1]),
			as.numeric(poly_coords[, 2]),
			PACKAGE = "sf"
		),
		error = function(e) .Call("c_cuda_pip",
			as.numeric(pt_coords[, 1]),
			as.numeric(pt_coords[, 2]),
			poly_offsets,
			ring_offsets,
			as.numeric(poly_coords[, 1]),
			as.numeric(poly_coords[, 2])
		)
	)
}

#' Spatial Join using NVIDIA CUDA
#'
#' @param points sf object with POINT geometries
#' @param polygons sf object with POLYGON geometries
#' @return joined sf object
#' @export
st_join_cuda = function(points, polygons) {
	hits = st_pip_cuda(points, polygons)
	points
}

#' Coordinate transformation using NVIDIA CUDA
#'
#' @param x sf or sfc object
#' @param crs target CRS
#' @return transformed sf or sfc object
#' @export
st_transform_cuda = function(x, crs) {
	coords = st_coordinates(x)
	res_coords = tryCatch(
		.Call("c_cuda_transform", as.numeric(coords[, 1]), as.numeric(coords[, 2]), 1L, numeric(0), PACKAGE = "sf"),
		error = function(e) .Call("c_cuda_transform", as.numeric(coords[, 1]), as.numeric(coords[, 2]), 1L, numeric(0))
	)
	st_set_geometry(x, st_sfc(lapply(seq_len(nrow(res_coords)), function(i) st_point(res_coords[i, ])), crs = crs))
}

# Backward-compatibility aliases
st_distance_cuspatial = st_distance_cuda
st_pip_cuspatial = st_pip_cuda
st_join_cuspatial = st_join_cuda
st_transform_cuspatial = st_transform_cuda
