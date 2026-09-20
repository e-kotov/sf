#pragma once

#include <Rinternals.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device query
SEXP c_cuda_device_count();

// Operations
SEXP c_cuda_distance(SEXP x_coords, SEXP y_coords, SEXP is_geodetic);
SEXP c_cuda_pip(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y);
SEXP c_cuda_quadtree_join(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y);
SEXP c_cuda_transform(SEXP in_x, SEXP in_y, SEXP proj_type, SEXP params);

#ifdef __cplusplus
}
#endif
