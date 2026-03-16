// 专门用于 ncu profiling，每个 kernel 只跑 1 次，无 warmup
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "nf4_dequant.cuh"
#include "nf4_dequant_v2.cuh"
#include "nf4_dequant_v4.cuh"
#include "nf4_dequant_v5.cuh"
#include "nf4_dequant_v7.cuh"   // ← 新增

static const int   N          = 1 << 24;
static const int   BLOCKSIZE  = 64;
static const int   BLOCKSIZE2 = 256;
static const float OFFSET     = 0.0f;

#define CUDA_CHECK(x) \
    do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
        fprintf(stderr,"CUDA error %s:%d %s\n",__FILE__,__LINE__, \
        cudaGetErrorString(_e)); exit(1); } } while(0)

int main(int argc, char** argv)
{
    // argv[1] 指定跑哪个版本: 1 2 4 5 7，默认全跑
    int target = argc > 1 ? atoi(argv[1]) : 0;

    int num_blocks = N / BLOCKSIZE;
    int num_groups = num_blocks / BLOCKSIZE2;

    srand(42);
    uint8_t* h_packed   = (uint8_t*)malloc(N / 2);
    uint8_t* h_absmax_q = (uint8_t*)malloc(num_blocks);
    float*   h_code2    = (float*)  malloc(256 * sizeof(float));
    float*   h_absmax2  = (float*)  malloc(num_groups * sizeof(float));
    for (int i = 0; i < N/2;        i++) h_packed[i]   = rand() & 0xFF;
    for (int i = 0; i < num_blocks;  i++) h_absmax_q[i] = rand() & 0xFF;
    for (int i = 0; i < 256;         i++) h_code2[i]   = (float)(rand()%200-100)/100.f;
    for (int i = 0; i < num_groups;  i++) h_absmax2[i] = (float)(rand()%100+1)  /100.f;

    uint8_t *d_packed, *d_absmax_q;
    float   *d_code2, *d_absmax2, *d_out;
    CUDA_CHECK(cudaMalloc(&d_packed,   N/2));
    CUDA_CHECK(cudaMalloc(&d_absmax_q, num_blocks));
    CUDA_CHECK(cudaMalloc(&d_code2,    256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_absmax2,  num_groups*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,      N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_packed,   h_packed,   N/2,                     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_absmax_q, h_absmax_q, num_blocks,              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_code2,    h_code2,    256*sizeof(float),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_absmax2,  h_absmax2,  num_groups*sizeof(float),cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;
    int grid_v1 = ((N/2)  + BLOCK-1) / BLOCK;
    int grid_v2 = (N/8    + BLOCK-1) / BLOCK;
    int grid_v4 = (N/16   + BLOCK-1) / BLOCK;
    int smem_v2 = (256 + num_groups) * sizeof(float);
    int smem_v5 = (BLOCK * 16 / BLOCKSIZE) * sizeof(float);
    // v7 grid 同 v4，每线程处理 16 元素，无 smem
    int grid_v7 = (N/16   + BLOCK-1) / BLOCK;  // ← 新增

    if (target == 0 || target == 1)
        nf4_dequant_kernel<<<grid_v1, BLOCK>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out, OFFSET, N, BLOCKSIZE, BLOCKSIZE2);

    if (target == 0 || target == 2)
        nf4_dequant_v2_kernel<<<grid_v2, BLOCK, smem_v2>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out, OFFSET, N, BLOCKSIZE, BLOCKSIZE2);

    if (target == 0 || target == 4)
        nf4_dequant_v4_kernel<<<grid_v4, BLOCK>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out, OFFSET, N, 6, 8);

    if (target == 0 || target == 5)
        nf4_dequant_v5_kernel<<<grid_v4, BLOCK, smem_v5>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out, OFFSET, N, 6, 8);

    if (target == 0 || target == 7)          // ← 新增
        nf4_dequant_v7_kernel<<<grid_v7, BLOCK>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out, OFFSET, N, 6, 8);

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_packed); cudaFree(d_absmax_q);
    cudaFree(d_code2);  cudaFree(d_absmax2); cudaFree(d_out);
    free(h_packed); free(h_absmax_q); free(h_code2); free(h_absmax2);
    return 0;
}