#include "nf4_gemv.cuh"

__constant__ float c_nf4_gemv[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
     0.0f,
     0.07958029955625534f,
     0.16093020141124725f,
     0.24611230194568634f,
     0.33791524171829224f,
     0.44070982933044434f,
     0.5626170039176941f,
     0.7229568362236023f,
     1.0f
};

#define WARP_SIZE       32
#define WARPS_PER_BLOCK 8

__global__ void nf4_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ code2,
    const float*   __restrict__ absmax2,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    float          offset,
    int            M,
    int            K,
    int            blocksize,
    int            blocksize2
)
{
    extern __shared__ float smem[];
    float* s_code2 = smem;          // [256]
    float* s_x     = smem + 256;    // [K]

    // 协作加载 code2 & x 到 shared memory
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        s_code2[i] = __ldg(&code2[i]);
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        s_x[i] = __ldg(&x[i]);
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;
    int row     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= M) return;

    int blocks_per_row  = K / blocksize;
    int global_bid_base = row * blocks_per_row;

    const uint8_t* row_packed   = packed   + (size_t)row * (K / 2);
    const uint8_t* row_absmax_q = absmax_q + (size_t)global_bid_base;

    float acc = 0.f;
    int half_K = K / 2;

    for (int byte_idx = lane; byte_idx < half_K; byte_idx += WARP_SIZE)
    {
        uint8_t b = __ldg(&row_packed[byte_idx]);

        int elem0 = byte_idx * 2;

        int global_bid = global_bid_base + elem0 / blocksize;
        int global_gid = global_bid / blocksize2;

        float scale = s_code2[__ldg(&row_absmax_q[elem0 / blocksize])]
                      * __ldg(&absmax2[global_gid])
                      + offset;

        // ✅ 修复：高4位 → elem0（偶数），低4位 → elem1（奇数）
        // 与 nf4_dequant_kernel 保持一致
        float w0 = c_nf4_gemv[(b >> 4) & 0xF] * scale;
        float w1 = c_nf4_gemv[ b       & 0xF] * scale;

        acc += w0 * s_x[elem0] + w1 * s_x[elem0 + 1];
    }

    // warp reduce sum
    #pragma unroll
    for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, delta);

    if (lane == 0)
        y[row] = acc;
}