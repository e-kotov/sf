#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

static id<MTLDevice> s_device = nil;
static id<MTLCommandQueue> s_queue = nil;
static id<MTLComputePipelineState> s_pipeline = nil;

static const char* SHADER_SRC = R"(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n1;
    uint n2;
};

kernel void distance_matrix(
    constant Params& params [[buffer(0)]],
    device const float2* pts1 [[buffer(1)]],
    device const float2* pts2 [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint2 id [[thread_position_in_grid]])
{
    if (id.x >= params.n1 || id.y >= params.n2) {
        return;
    }
    float2 p1 = pts1[id.x];
    float2 p2 = pts2[id.y];
    out[id.x + id.y * params.n1] = distance(p1, p2);
}
)";

extern "C" int c_metal_distance_matrix(
    const int* n1_ptr,
    const float* pts1_ptr,
    const int* n2_ptr,
    const float* pts2_ptr,
    float* out_ptr)
{
    @autoreleasepool {
        int n1 = *n1_ptr;
        int n2 = *n2_ptr;
        if (n1 == 0 || n2 == 0) return 0;

        if (!s_device) {
            s_device = MTLCreateSystemDefaultDevice();
            if (!s_device) return -1;
            s_queue = [s_device newCommandQueue];

            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:SHADER_SRC];
            id<MTLLibrary> lib = [s_device newLibraryWithSource:source options:nil error:&error];
            if (!lib) {
                NSLog(@"Metal shader compile error: %@", error);
                return -2;
            }

            id<MTLFunction> fn = [lib newFunctionWithName:@"distance_matrix"];
            s_pipeline = [s_device newComputePipelineStateWithFunction:fn error:&error];
            if (!s_pipeline) {
                NSLog(@"Metal pipeline error: %@", error);
                return -3;
            }
        }

        struct Params { uint32_t n1, n2; } params = { (uint32_t)n1, (uint32_t)n2 };

        id<MTLBuffer> buf_params = [s_device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
        
        // Zero-copy: wrap pointers directly into shared unified memory
        // Note: newBufferWithBytesNoCopy requires page-aligned memory or fallback to newBufferWithBytes
        id<MTLBuffer> buf_pts1 = nil;
        id<MTLBuffer> buf_pts2 = nil;
        id<MTLBuffer> buf_out = nil;

        uintptr_t p1_addr = (uintptr_t)pts1_ptr;
        uintptr_t p2_addr = (uintptr_t)pts2_ptr;
        uintptr_t out_addr = (uintptr_t)out_ptr;
        size_t page_mask = getpagesize() - 1;

        if ((p1_addr & page_mask) == 0 && (p2_addr & page_mask) == 0 && (out_addr & page_mask) == 0) {
            buf_pts1 = [s_device newBufferWithBytesNoCopy:(void*)pts1_ptr length:2 * n1 * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
            buf_pts2 = [s_device newBufferWithBytesNoCopy:(void*)pts2_ptr length:2 * n2 * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
            buf_out = [s_device newBufferWithBytesNoCopy:(void*)out_ptr length:(size_t)n1 * n2 * sizeof(float) options:MTLResourceStorageModeShared deallocator:nil];
        } else {
            buf_pts1 = [s_device newBufferWithBytes:pts1_ptr length:2 * n1 * sizeof(float) options:MTLResourceStorageModeShared];
            buf_pts2 = [s_device newBufferWithBytes:pts2_ptr length:2 * n2 * sizeof(float) options:MTLResourceStorageModeShared];
            buf_out = [s_device newBufferWithLength:(size_t)n1 * n2 * sizeof(float) options:MTLResourceStorageModeShared];
        }

        id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:s_pipeline];
        [enc setBuffer:buf_params offset:0 atIndex:0];
        [enc setBuffer:buf_pts1 offset:0 atIndex:1];
        [enc setBuffer:buf_pts2 offset:0 atIndex:2];
        [enc setBuffer:buf_out offset:0 atIndex:3];

        MTLSize gridSize = MTLSizeMake(n1, n2, 1);
        NSUInteger w = 16;
        NSUInteger h = 16;
        MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        if ((out_addr & page_mask) != 0) {
            memcpy(out_ptr, [buf_out contents], (size_t)n1 * n2 * sizeof(float));
        }

        return 0;
    }
}
