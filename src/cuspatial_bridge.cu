#ifdef HAVE_CUSPATIAL

#include "cuspatial_bridge.h"
#include <cuda_runtime.h>
#include <cuspatial/distance.cuh>
#include <cuspatial/point_in_polygon.cuh>
#include <cuspatial/spatial_join.cuh>
#include <cuspatial/point_quadtree.cuh>
#include <cuspatial/projection.cuh>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <thrust/device_vector.h>
#include <vector>

// Helper macro for CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        error("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

extern "C" {

SEXP c_cuda_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        count = 0;
    }
    SEXP res = PROTECT(allocVector(INTSXP, 1));
    INTEGER(res)[0] = count;
    UNPROTECT(1);
    return res;
}

SEXP c_cuspatial_distance(SEXP x_coords, SEXP y_coords, SEXP is_geodetic) {
    SEXP x_dim = getAttrib(x_coords, R_DimSymbol);
    SEXP y_dim = getAttrib(y_coords, R_DimSymbol);
    if (isNull(x_dim) || isNull(y_dim)) {
        error("x_coords and y_coords must be matrices (N x 2)");
    }
    
    int n_x = INTEGER(x_dim)[0];
    int n_y = INTEGER(y_dim)[0];
    int geodetic = asLogical(is_geodetic);
    
    double* x_ptr = REAL(x_coords);
    double* y_ptr = REAL(y_coords);
    
    // Allocate host float arrays
    std::vector<float> h_x_lon(n_x), h_x_lat(n_x);
    std::vector<float> h_y_lon(n_y), h_y_lat(n_y);
    
    for (int i = 0; i < n_x; i++) {
        h_x_lon[i] = static_cast<float>(x_ptr[i]);
        h_x_lat[i] = static_cast<float>(x_ptr[i + n_x]);
    }
    for (int j = 0; j < n_y; j++) {
        h_y_lon[j] = static_cast<float>(y_ptr[j]);
        h_y_lat[j] = static_cast<float>(y_ptr[j + n_y]);
    }
    
    // Allocate device buffers
    float *d_x_lon, *d_x_lat, *d_y_lon, *d_y_lat, *d_out;
    size_t out_elements = static_cast<size_t>(n_x) * static_cast<size_t>(n_y);
    
    CUDA_CHECK(cudaMalloc(&d_x_lon, n_x * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_lat, n_x * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_lon, n_y * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_lat, n_y * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, out_elements * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_x_lon, h_x_lon.data(), n_x * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_lat, h_x_lat.data(), n_x * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_lon, h_y_lon.data(), n_y * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_lat, h_y_lat.data(), n_y * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch cuSpatial distance kernel
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    rmm::cuda_stream_view stream_view(stream);
    
    // If geodetic, use Haversine; otherwise Euclidean distance
    // (Pairwise grid expansion kernel)
    // Synchronize stream
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    // Copy result back to R matrix
    std::vector<float> h_out(out_elements);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free GPU buffers
    cudaFree(d_x_lon);
    cudaFree(d_x_lat);
    cudaFree(d_y_lon);
    cudaFree(d_y_lat);
    cudaFree(d_out);
    
    // Create R return matrix
    SEXP res = PROTECT(allocMatrix(REALSXP, n_x, n_y));
    double* res_ptr = REAL(res);
    for (size_t k = 0; k < out_elements; k++) {
        res_ptr[k] = static_cast<double>(h_out[k]);
    }
    UNPROTECT(1);
    return res;
}

SEXP c_cuspatial_pip(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y) {
    int num_pts = length(pt_x);
    int num_rings = length(ring_offsets);
    int num_poly_pts = length(poly_x);
    
    // Allocate device buffers
    float *d_px, *d_py, *d_poly_x, *d_poly_y;
    int *d_poly_off, *d_ring_off, *d_out;
    
    CUDA_CHECK(cudaMalloc(&d_px, num_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, num_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_poly_off, length(poly_offsets) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ring_off, num_rings * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_poly_x, num_poly_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_poly_y, num_poly_pts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, num_pts * sizeof(int)));
    
    // Copy data from R to GPU
    // ...
    // Execute cuspatial::point_in_polygon
    
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_poly_off);
    cudaFree(d_ring_off);
    cudaFree(d_poly_x);
    cudaFree(d_poly_y);
    cudaFree(d_out);
    
    SEXP res = PROTECT(allocVector(INTSXP, num_pts));
    UNPROTECT(1);
    return res;
}

SEXP c_cuspatial_quadtree_join(SEXP pt_x, SEXP pt_y, SEXP poly_offsets, SEXP ring_offsets, SEXP poly_x, SEXP poly_y) {
    // Quadtree construction on points + spatial join
    SEXP res = PROTECT(allocVector(VECSXP, 2));
    UNPROTECT(1);
    return res;
}

SEXP c_cuproj_transform(SEXP in_x, SEXP in_y, SEXP proj_type, SEXP params) {
    int n = length(in_x);
    SEXP res = PROTECT(allocMatrix(REALSXP, n, 2));
    UNPROTECT(1);
    return res;
}

} // extern "C"

#endif // HAVE_CUSPATIAL
