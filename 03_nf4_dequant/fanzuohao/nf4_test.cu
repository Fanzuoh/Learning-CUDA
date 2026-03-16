#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "nf4_dequant.cuh"
#include "nf4_dequant_v2.cuh"
#include "nf4_dequant_v4.cuh"
#include "nf4_dequant_v5.cuh"
#include "nf4_dequant_v7.cuh"   // ← 1. 新增头文件

static const int   N          = 1 << 24;
static const int   BLOCKSIZE  = 64;
static const int   BLOCKSIZE2 = 256;
static const float OFFSET     = 0.0f;
static const int   WARMUP     = 10;
static const int   REPEAT     = 100;

#define CUDA_CHECK(x)                                                      \
    do {                                                                    \
        cudaError_t _e = (x);                                              \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

template<typename F>
static float time_kernel(cudaEvent_t start, cudaEvent_t stop,
                          int repeat, F launch_fn)
{
    for (int i = 0; i < WARMUP; i++) launch_fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) launch_fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / repeat;
}

static void check_fp32(const float* ref, const float* got, int n,
                        const char* name, float tol = 1e-5f)
{
    int err = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(ref[i] - got[i]) > tol) {
            if (err < 5)
                printf("  [%s] MISMATCH at %d: ref=%.6f got=%.6f\n",
                       name, i, ref[i], got[i]);
            err++;
        }
    }
    if (err == 0) printf("  [%s] PASS ✓\n", name);
    else          printf("  [%s] FAIL ✗  %d / %d errors\n", name, err, n);
}

int main()
{
    int num_blocks = N / BLOCKSIZE;
    int num_groups = num_blocks / BLOCKSIZE2;

    printf("N=%d  blocksize=%d  blocksize2=%d\n", N, BLOCKSIZE, BLOCKSIZE2);
    printf("num_blocks=%d  num_groups=%d\n", num_blocks, num_groups);

    srand(42);
    uint8_t* h_packed   = (uint8_t*)malloc(N / 2);
    uint8_t* h_absmax_q = (uint8_t*)malloc(num_blocks);
    float*   h_code2    = (float*)  malloc(256 * sizeof(float));
    float*   h_absmax2  = (float*)  malloc(num_groups * sizeof(float));

    for (int i = 0; i < N / 2;     i++) h_packed[i]   = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < num_blocks; i++) h_absmax_q[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < 256;        i++) h_code2[i]   = (float)(rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < num_groups; i++) h_absmax2[i] = (float)(rand() % 100 + 1)   / 100.0f;

    uint8_t *d_packed, *d_absmax_q;
    float   *d_code2, *d_absmax2;
    float   *d_out_v1, *d_out_v2, *d_out_v4, *d_out_v5, *d_out_v7; // ← 2. 新增 v7

    CUDA_CHECK(cudaMalloc(&d_packed,   N / 2));
    CUDA_CHECK(cudaMalloc(&d_absmax_q, num_blocks));
    CUDA_CHECK(cudaMalloc(&d_code2,    256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_absmax2,  num_groups * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_v1,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_v2,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_v4,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_v5,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_v7,   N * sizeof(float)));           // ← 2. 新增 v7

    CUDA_CHECK(cudaMemcpy(d_packed,   h_packed,   N / 2,                     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_absmax_q, h_absmax_q, num_blocks,                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_code2,    h_code2,    256 * sizeof(float),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_absmax2,  h_absmax2,  num_groups * sizeof(float),cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;
    int grid_v1  = ((N / 2)  + BLOCK - 1) / BLOCK;
    int grid_v2  = (N / 8    + BLOCK - 1) / BLOCK;
    int smem_v2  = (256 + num_groups) * sizeof(float);
    int grid_v4  = (N / 16   + BLOCK - 1) / BLOCK;
    int grid_v5  = (N / 16   + BLOCK - 1) / BLOCK;
    int smem_v5  = (BLOCK * 16 / BLOCKSIZE) * sizeof(float);
    int grid_v7  = (N / 16   + BLOCK - 1) / BLOCK;  // ← 2. 同 v4，每线程 16 元素

    // ── 正确性验证 ──
    printf("\n=== Correctness Check ===\n");

    nf4_dequant_kernel<<<grid_v1, BLOCK>>>(
        d_packed, d_absmax_q, d_code2, d_absmax2,
        d_out_v1, OFFSET, N, BLOCKSIZE, BLOCKSIZE2);
    CUDA_CHECK(cudaDeviceSynchronize());

    nf4_dequant_v2_kernel<<<grid_v2, BLOCK, smem_v2>>>(
        d_packed, d_absmax_q, d_code2, d_absmax2,
        d_out_v2, OFFSET, N, BLOCKSIZE, BLOCKSIZE2);
    CUDA_CHECK(cudaDeviceSynchronize());

    nf4_dequant_v4_kernel<<<grid_v4, BLOCK>>>(
        d_packed, d_absmax_q, d_code2, d_absmax2,
        d_out_v4, OFFSET, N, 6, 8);
    CUDA_CHECK(cudaDeviceSynchronize());

    nf4_dequant_v5_kernel<<<grid_v5, BLOCK, smem_v5>>>(
        d_packed, d_absmax_q, d_code2, d_absmax2,
        d_out_v5, OFFSET, N, 6, 8);
    CUDA_CHECK(cudaDeviceSynchronize());

    nf4_dequant_v7_kernel<<<grid_v7, BLOCK>>>(   // ← 3. 新增 v7 correctness
        d_packed, d_absmax_q, d_code2, d_absmax2,
        d_out_v7, OFFSET, N, 6, 8);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_out_v1 = (float*)malloc(N * sizeof(float));
    float* h_out_v2 = (float*)malloc(N * sizeof(float));
    float* h_out_v4 = (float*)malloc(N * sizeof(float));
    float* h_out_v5 = (float*)malloc(N * sizeof(float));
    float* h_out_v7 = (float*)malloc(N * sizeof(float));  // ← 3. 新增 v7

    CUDA_CHECK(cudaMemcpy(h_out_v1, d_out_v1, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_v2, d_out_v2, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_v4, d_out_v4, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_v5, d_out_v5, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_v7, d_out_v7, N*sizeof(float), cudaMemcpyDeviceToHost)); // ← 3.

    check_fp32(h_out_v1, h_out_v2, N, "v2 vs v1");
    check_fp32(h_out_v1, h_out_v4, N, "v4 vs v1");
    check_fp32(h_out_v1, h_out_v5, N, "v5 vs v1");
    check_fp32(h_out_v1, h_out_v7, N, "v7 vs v1");  // ← 3. 新增验证

    // ── 性能测试 ──
    printf("\n=== Performance (avg over %d runs) ===\n", REPEAT);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float t_v1 = time_kernel(start, stop, REPEAT, [&](){
        nf4_dequant_kernel<<<grid_v1, BLOCK>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out_v1, OFFSET, N, BLOCKSIZE, BLOCKSIZE2);
    });
    float t_v2 = time_kernel(start, stop, REPEAT, [&](){
        nf4_dequant_v2_kernel<<<grid_v2, BLOCK, smem_v2>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out_v2, OFFSET, N, BLOCKSIZE, BLOCKSIZE2);
    });
    float t_v4 = time_kernel(start, stop, REPEAT, [&](){
        nf4_dequant_v4_kernel<<<grid_v4, BLOCK>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out_v4, OFFSET, N, 6, 8);
    });
    float t_v5 = time_kernel(start, stop, REPEAT, [&](){
        nf4_dequant_v5_kernel<<<grid_v5, BLOCK, smem_v5>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out_v5, OFFSET, N, 6, 8);
    });
    float t_v7 = time_kernel(start, stop, REPEAT, [&](){ // ← 4. 新增 v7 benchmark
        nf4_dequant_v7_kernel<<<grid_v7, BLOCK>>>(
            d_packed, d_absmax_q, d_code2, d_absmax2,
            d_out_v7, OFFSET, N, 6, 8);
    });

    double bytes = (double)(N/2) + num_blocks
                 + 256*sizeof(float) + num_groups*sizeof(float)
                 + (double)N*sizeof(float);

    printf("  v1 : %7.3f ms   BW = %6.1f GB/s\n", t_v1, bytes / t_v1 / 1e6);
    printf("  v2 : %7.3f ms   BW = %6.1f GB/s\n", t_v2, bytes / t_v2 / 1e6);
    printf("  v4 : %7.3f ms   BW = %6.1f GB/s\n", t_v4, bytes / t_v4 / 1e6);
    printf("  v5 : %7.3f ms   BW = %6.1f GB/s\n", t_v5, bytes / t_v5 / 1e6);
    printf("  v7 : %7.3f ms   BW = %6.1f GB/s\n", t_v7, bytes / t_v7 / 1e6); // ← 4.
    printf("\n  speedup v4/v1 = %.2fx\n", t_v1 / t_v4);
    printf("  speedup v5/v1 = %.2fx\n",  t_v1 / t_v5);
    printf("  speedup v7/v1 = %.2fx\n",  t_v1 / t_v7);  // ← 4.
    printf("  speedup v7/v4 = %.2fx\n",  t_v4 / t_v7);  // ← 4. 关键对比

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_packed);   cudaFree(d_absmax_q);
    cudaFree(d_code2);    cudaFree(d_absmax2);
    cudaFree(d_out_v1);   cudaFree(d_out_v2);
    cudaFree(d_out_v4);   cudaFree(d_out_v5);
    cudaFree(d_out_v7);                                   // ← 4.
    free(h_packed);   free(h_absmax_q);
    free(h_code2);    free(h_absmax2);
    free(h_out_v1);   free(h_out_v2);
    free(h_out_v4);   free(h_out_v5);
    free(h_out_v7);                                       // ← 4.
    return 0;
}