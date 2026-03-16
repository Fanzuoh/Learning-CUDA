// /home/fanzuohao/kernel_test/nf4_dequant_v2.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// 优化版：每线程处理 8 个元素（4个packed byte），float4 向量写出
__global__ void nf4_dequant_v2_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ code2,
    const float*   __restrict__ absmax2,
    float*         __restrict__ out,
    float          offset,
    int            N,
    int            blocksize,
    int            blocksize2
);
