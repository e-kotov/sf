#include "cuspatial_bridge.h"
#include <Rinternals.h>

#ifndef HAVE_CUSPATIAL

SEXP c_cuda_device_count() {
    SEXP res = PROTECT(allocVector(INTSXP, 1));
    INTEGER(res)[0] = 0;
    UNPROTECT(1);
    return res;
}

SEXP c_cuspatial_distance(SEXP x_coords, SEXP y_coords, SEXP is_geodetic) {
    error("cuSpatial is not compiled or available on this system. cuSpatial requires an NVIDIA GPU with CUDA. On macOS, please use the Metal backend via sf_use_gpu().");
    return R_NilValue;
}

SEXP c_cuspatial_pip(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y) {
    error("cuSpatial is not compiled or available on this system. cuSpatial requires an NVIDIA GPU with CUDA. On macOS, please use the Metal or CPU backend.");
    return R_NilValue;
}

SEXP c_cuspatial_quadtree_join(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y) {
    error("cuSpatial is not compiled or available on this system. cuSpatial requires an NVIDIA GPU with CUDA.");
    return R_NilValue;
}

SEXP c_cuproj_transform(SEXP in_x, SEXP in_y, SEXP proj_type, SEXP params) {
    error("cuProj is not compiled or available on this system. cuProj requires an NVIDIA GPU with CUDA.");
    return R_NilValue;
}

#endif
