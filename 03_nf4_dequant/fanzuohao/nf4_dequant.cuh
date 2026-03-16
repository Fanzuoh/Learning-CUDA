#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// v3: BF16输出 + uint32_t向量化写入 + FP16 code2/absmax2
// 每个线程处理 1 个 packed byte → 2 个 BF16 输出元素（打包成 1 个 uint32_t 写入）
// grid : ((N/2) + BLOCK - 1) / BLOCK
// block: BLOCK (建议256)
__global__ void nf4_dequant_kernel_v3(
    const uint8_t*  __restrict__ packed,      // [N/2]
    const uint8_t*  __restrict__ absmax_q,    // [num_blocks]
    const __half*   __restrict__ code2,       // [256]  FP16
    const __half*   __restrict__ absmax2,     // [num_groups]  FP16
    uint32_t*       __restrict__ out,         // [N/2]  每个uint32_t存2个BF16
    float           offset,
    int             N,
    int             blocksize,
    int             blocksize2
);
__global__ void nf4_dequant_kernel(
    const uint8_t*  __restrict__ packed,
    const uint8_t*  __restrict__ absmax_q,
    const float*    __restrict__ code2,       // FP32
    const float*    __restrict__ absmax2,     // FP32
    float*          __restrict__ out,         // FP32 输出
    float           offset,
    int             N,
    int             blocksize,
    int             blocksize2
);
