#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// upload_constants_v4 已废弃，不再需要
// code2 / absmax2 直接作为 kernel 参数传入（设备指针）

// v4 kernel：每线程处理 16 个元素（8 个 packed byte）
// grid : (N/16 + BLOCK - 1) / BLOCK
// block: 256
__global__ void nf4_dequant_v4_kernel(
    const uint8_t* __restrict__ packed,     // [N/2]
    const uint8_t* __restrict__ absmax_q,   // [N/blocksize]
    const float*   __restrict__ code2,      // [256]  设备指针，__ldg 读
    const float*   __restrict__ absmax2,    // [num_groups] 设备指针，__ldg 读
    float*         __restrict__ out,        // [N]
    float          offset,
    int            N,
    int            log2_blocksize,          // blocksize=64  → 传 6
    int            log2_blocksize2          // blocksize2=256 → 传 8（注意：是 blocksize/blocksize2 的 log2）
);