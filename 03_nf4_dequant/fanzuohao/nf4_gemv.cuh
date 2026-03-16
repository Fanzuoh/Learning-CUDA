// /home/fanzuohao/kernel_test/nf4_gemv.cuh
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Fused NF4 dequant + GEMV
// W: [M, K] NF4 packed → packed[M * K/2]，行主序
// x: [K] float32
// y: [M] float32  (输出)
//
// 参数与 dequant kernel 一致：
//   absmax_q : [M * K / blocksize]   每个 block 的量化 absmax index
//   code2    : [256]                 二级 codebook
//   absmax2  : [M * K / blocksize / blocksize2]  二级 absmax
//   offset   : dequant offset
//   blocksize  : 64  (一级 block 大小)
//   blocksize2 : 256 (二级 block 大小)
__global__ void nf4_gemv_kernel(
    const uint8_t* __restrict__ packed,    // [M * K/2]
    const uint8_t* __restrict__ absmax_q,  // [M * K / blocksize]
    const float*   __restrict__ code2,     // [256]
    const float*   __restrict__ absmax2,   // [M * K / blocksize / blocksize2]
    const float*   __restrict__ x,         // [K]
    float*         __restrict__ y,         // [M]
    float          offset,
    int            M,
    int            K,
    int            blocksize,
    int            blocksize2
);