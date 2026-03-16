#include "nf4_dequant_v7.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// NF4 表放寄存器，彻底消除 MIO 查表
// 编译器会把 nf4_reg[idx] 展开成 if-else 树或 LUT in RF
__global__ void nf4_dequant_v7_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ code2,
    const float*   __restrict__ absmax2,
    float*         __restrict__ out,
    float          offset,
    int            N,
    int            log2_blocksize,
    int            log2_blocksize2
)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    int base_elem = tid << 4;
    if (base_elem >= N) return;

    // ── NF4 表：寄存器数组，编译器展开为 if-else 树，0 MIO ──
    const float nf4[16] = {
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

    // ── scale：寄存器缓存，最多 2 个 ──
    int bid0 = base_elem >> log2_blocksize;
    int bid1 = (base_elem + 15) >> log2_blocksize;

    auto load_scale = [&](int bid) -> float {
        int gid = bid >> log2_blocksize2;
        return __ldg(&code2[__ldg(&absmax_q[bid])]) * __ldg(&absmax2[gid]) + offset;
    };

    float s0 = load_scale(bid0);
    float s1 = (bid1 != bid0) ? load_scale(bid1) : s0;
    int boundary = ((bid0 + 1) << log2_blocksize) - base_elem;

    // ── 64-bit 向量读取 packed 数据 ──
    int  base_byte = tid << 3;
    uint2 raw = *reinterpret_cast<const uint2*>(&packed[base_byte]);
    uint8_t b[8];
    b[0] = (raw.x >>  0) & 0xFF;  b[1] = (raw.x >>  8) & 0xFF;
    b[2] = (raw.x >> 16) & 0xFF;  b[3] = (raw.x >> 24) & 0xFF;
    b[4] = (raw.y >>  0) & 0xFF;  b[5] = (raw.y >>  8) & 0xFF;
    b[6] = (raw.y >> 16) & 0xFF;  b[7] = (raw.y >> 24) & 0xFF;

    // ── 展开 16 次查表+乘法，全在寄存器 ──
#define SCALE(i) ((i) < boundary ? s0 : s1)
#define ELEM(bi, sh) nf4[(b[bi] >> (sh)) & 0xF]

    float4 r0, r1, r2, r3;
    r0.x = ELEM(0,4) * SCALE( 0);  r0.y = ELEM(0,0) * SCALE( 1);
    r0.z = ELEM(1,4) * SCALE( 2);  r0.w = ELEM(1,0) * SCALE( 3);
    r1.x = ELEM(2,4) * SCALE( 4);  r1.y = ELEM(2,0) * SCALE( 5);
    r1.z = ELEM(3,4) * SCALE( 6);  r1.w = ELEM(3,0) * SCALE( 7);
    r2.x = ELEM(4,4) * SCALE( 8);  r2.y = ELEM(4,0) * SCALE( 9);
    r2.z = ELEM(5,4) * SCALE(10);  r2.w = ELEM(5,0) * SCALE(11);
    r3.x = ELEM(6,4) * SCALE(12);  r3.y = ELEM(6,0) * SCALE(13);
    r3.z = ELEM(7,4) * SCALE(14);  r3.w = ELEM(7,0) * SCALE(15);
#undef SCALE
#undef ELEM

    // ── 128-bit × 4 向量写出 ──
    float4* out4 = reinterpret_cast<float4*>(out) + (base_elem >> 2);
    out4[0] = r0;  out4[1] = r1;  out4[2] = r2;  out4[3] = r3;
}