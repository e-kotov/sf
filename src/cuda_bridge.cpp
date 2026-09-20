#include "cuda_bridge.h"
#include <Rinternals.h>

#ifndef HAVE_CUDA

SEXP c_cuda_device_count() {
    SEXP res = PROTECT(allocVector(INTSXP, 1));
    INTEGER(res)[0] = 0;
    UNPROTECT(1);
    return res;
}

SEXP c_cuda_distance(SEXP x_coords, SEXP y_coords, SEXP is_geodetic) {
    error("CUDA backend is not compiled or available on this system. CUDA requires an NVIDIA GPU. On macOS, please use the Metal backend via sf_use_gpu().");
    return R_NilValue;
}

SEXP c_cuda_pip(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y) {
    error("CUDA backend is not compiled or available on this system. CUDA requires an NVIDIA GPU. On macOS, please use the Metal or CPU backend.");
    return R_NilValue;
}

SEXP c_cuda_quadtree_join(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y) {
    error("CUDA backend is not compiled or available on this system. CUDA requires an NVIDIA GPU.");
    return R_NilValue;
}

SEXP c_cuda_transform(SEXP in_x, SEXP in_y, SEXP proj_type, SEXP params) {
    error("CUDA backend is not compiled or available on this system. CUDA requires an NVIDIA GPU.");
    return R_NilValue;
}

#endif
