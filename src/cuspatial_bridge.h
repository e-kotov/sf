#pragma once

#include <Rinternals.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device query
SEXP c_cuda_device_count();

// Operations
SEXP c_cuspatial_distance(SEXP x_coords, SEXP y_coords, SEXP is_geodetic);
SEXP c_cuspatial_pip(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y);
SEXP c_cuspatial_quadtree_join(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y);
SEXP c_cuproj_transform(SEXP in_x, SEXP in_y, SEXP proj_type, SEXP params);

#ifdef __cplusplus
}
#endif
