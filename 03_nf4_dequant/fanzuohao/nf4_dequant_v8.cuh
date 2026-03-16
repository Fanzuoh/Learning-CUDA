#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>

extern __global__ void nf4_dequant_v8_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ code2,
    const float*   __restrict__ absmax2,
    __half*        __restrict__ out,
    float          offset,
    int            N,
    int            log2_blocksize,
    int            log2_blocksize2
);