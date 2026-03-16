#include "nf4_dequant_v8.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ── NF4 码表：constant memory，warp 内广播，0 额外延迟 ──
__constant__ float c_nf4_v8[16] = {
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

__global__ void nf4_dequant_v8_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ code2,
    const float*   __restrict__ absmax2,
    __half*        __restrict__ out,
    float          offset,
    int            N,
    int            log2_blocksize,
    int            log2_blocksize2
)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    int base_elem = tid << 4;
    if (base_elem >= N) return;

    // ── 1. scale 预计算：最多 2 个 block，只做 2 次 __ldg ──
    int bid0 = base_elem >> log2_blocksize;
    int bid1 = (base_elem + 15) >> log2_blocksize;

    auto load_scale = [&](int bid) -> float {
        int gid = bid >> log2_blocksize2;
        return __ldg(&code2[__ldg(&absmax_q[bid])]) * __ldg(&absmax2[gid]) + offset;
    };

    float s0 = load_scale(bid0);
    float s1 = (bid1 != bid0) ? load_scale(bid1) : s0;
    int boundary = ((bid0 + 1) << log2_blocksize) - base_elem;

    // ── 2. 64-bit 向量读取 8 个 packed byte ──
    int   base_byte = tid << 3;
    uint2 raw = *reinterpret_cast<const uint2*>(&packed[base_byte]);
    uint8_t b[8];
    b[0] = (raw.x >>  0) & 0xFF;  b[1] = (raw.x >>  8) & 0xFF;
    b[2] = (raw.x >> 16) & 0xFF;  b[3] = (raw.x >> 24) & 0xFF;
    b[4] = (raw.y >>  0) & 0xFF;  b[5] = (raw.y >>  8) & 0xFF;
    b[6] = (raw.y >> 16) & 0xFF;  b[7] = (raw.y >> 24) & 0xFF;

    // ── 3. 展开 16 次查表+乘法，转换为 half ──
#define SCALE(i) ((i) < boundary ? s0 : s1)
#define ELEM(bi, sh) c_nf4_v8[(b[bi] >> (sh)) & 0xF]
#define CONV(bi, sh, i) __float2half_rn(ELEM(bi, sh) * SCALE(i))

    half2 h[8];
    h[0] = __halves2half2(CONV(0,4, 0), CONV(0,0, 1));
    h[1] = __halves2half2(CONV(1,4, 2), CONV(1,0, 3));
    h[2] = __halves2half2(CONV(2,4, 4), CONV(2,0, 5));
    h[3] = __halves2half2(CONV(3,4, 6), CONV(3,0, 7));
    h[4] = __halves2half2(CONV(4,4, 8), CONV(4,0, 9));
    h[5] = __halves2half2(CONV(5,4,10), CONV(5,0,11));
    h[6] = __halves2half2(CONV(6,4,12), CONV(6,0,13));
    h[7] = __halves2half2(CONV(7,4,14), CONV(7,0,15));
#undef CONV
#undef ELEM
#undef SCALE

    // ── 4. 2× 128-bit 向量写出 ──
    // half2[8] = 32 bytes = 2 × float4
    float4* out4 = reinterpret_cast<float4*>(out) + (base_elem >> 3);
    out4[0] = *reinterpret_cast<float4*>(&h[0]);
    out4[1] = *reinterpret_cast<float4*>(&h[4]);
}
