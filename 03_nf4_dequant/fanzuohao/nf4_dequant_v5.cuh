#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// v5 kernel：预计算 scale 到 shared memory，消除 global mem 依赖链
//
// grid : (N/16 + BLOCK - 1) / BLOCK
// block: 256
// smem : scales_per_block * sizeof(float)
//        = (BLOCK * 16 / blocksize) * 4
//        = (256 * 16 / 64) * 4 = 256 bytes
__global__ void nf4_dequant_v5_kernel(
    const uint8_t* __restrict__ packed,     // [N/2]
    const uint8_t* __restrict__ absmax_q,   // [N/blocksize]
    const float*   __restrict__ code2,      // [256]
    const float*   __restrict__ absmax2,    // [num_groups]
    float*         __restrict__ out,        // [N]
    float          offset,
    int            N,
    int            log2_blocksize,          // blocksize=64  → 6
    int            log2_blocksize2          // blocksize2=256 → 8
);