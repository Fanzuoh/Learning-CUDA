#include "nf4_dequant_v4.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// ── NF4 码表：只有 16 个 float，所有线程读同一个索引 → 广播命中 ✅
__constant__ float c_nf4_v4[16] = {
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

// ────────────────────────────────────────────────────────────────
//  v4 kernel（修复版）
//
//  关键改动：
//  - code2 / absmax2 从 constant memory 移到 global memory 参数
//  - 用 __ldg() 走 read-only (texture) cache
//    → 同 warp 内相邻线程访问相邻地址，L1 cache line 命中
//  - c_nf4_v4 保留 constant（16个float，warp内所有线程读同一idx → 广播）
//
//  每线程处理 8 packed bytes = 16 NF4 元素
//  读：1× uint2 load  (64-bit)
//  写：4× float4 store (128-bit × 4)
// ────────────────────────────────────────────────────────────────
__global__ void nf4_dequant_v4_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ code2,     // ← global, __ldg
    const float*   __restrict__ absmax2,   // ← global, __ldg
    float*         __restrict__ out,
    float          offset,
    int            N,
    int            log2_blocksize,
    int            log2_blocksize2
)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    int base_elem = tid << 4;          // tid * 16
    if (base_elem >= N) return;

    int base_byte = tid << 3;          // tid * 8

    // ── 1. 64-bit 向量读取 8 个 packed byte ──
    uint2 raw = *reinterpret_cast<const uint2*>(&packed[base_byte]);
    uint8_t b[8];
    b[0] = (raw.x >>  0) & 0xFF;
    b[1] = (raw.x >>  8) & 0xFF;
    b[2] = (raw.x >> 16) & 0xFF;
    b[3] = (raw.x >> 24) & 0xFF;
    b[4] = (raw.y >>  0) & 0xFF;
    b[5] = (raw.y >>  8) & 0xFF;
    b[6] = (raw.y >> 16) & 0xFF;
    b[7] = (raw.y >> 24) & 0xFF;

    // ── 2. scale 计算 ──
    // 同一线程的 16 个元素最多跨 2 个 blocksize=64 的 block
    // 同一 warp 的 32 线程跨 8 个 block（连续），L1 cache line = 128B = 32 floats
    // absmax_q 访问：32线程 × 1 byte = 32B → 1 cache line ✅
    // absmax2  访问：最多 8 个 gid  × 4B = 32B → 1 cache line ✅
    // code2    访问：最多 8 个不同idx × 4B = 32B → 1 cache line ✅
    auto get_scale = [&](int elem_idx) -> float {
        int bid = elem_idx >> log2_blocksize;
        int gid = bid      >> log2_blocksize2;
        // __ldg: 走 read-only cache，不污染 L1 data cache
        return __ldg(&code2[__ldg(&absmax_q[bid])]) * __ldg(&absmax2[gid]) + offset;
    };

    // ── 3. 展开计算 16 个输出 ──
    float4 r0, r1, r2, r3;

    r0.x = c_nf4_v4[(b[0] >> 4) & 0xF] * get_scale(base_elem +  0);
    r0.y = c_nf4_v4[ b[0]       & 0xF] * get_scale(base_elem +  1);
    r0.z = c_nf4_v4[(b[1] >> 4) & 0xF] * get_scale(base_elem +  2);
    r0.w = c_nf4_v4[ b[1]       & 0xF] * get_scale(base_elem +  3);

    r1.x = c_nf4_v4[(b[2] >> 4) & 0xF] * get_scale(base_elem +  4);
    r1.y = c_nf4_v4[ b[2]       & 0xF] * get_scale(base_elem +  5);
    r1.z = c_nf4_v4[(b[3] >> 4) & 0xF] * get_scale(base_elem +  6);
    r1.w = c_nf4_v4[ b[3]       & 0xF] * get_scale(base_elem +  7);

    r2.x = c_nf4_v4[(b[4] >> 4) & 0xF] * get_scale(base_elem +  8);
    r2.y = c_nf4_v4[ b[4]       & 0xF] * get_scale(base_elem +  9);
    r2.z = c_nf4_v4[(b[5] >> 4) & 0xF] * get_scale(base_elem + 10);
    r2.w = c_nf4_v4[ b[5]       & 0xF] * get_scale(base_elem + 11);

    r3.x = c_nf4_v4[(b[6] >> 4) & 0xF] * get_scale(base_elem + 12);
    r3.y = c_nf4_v4[ b[6]       & 0xF] * get_scale(base_elem + 13);
    r3.z = c_nf4_v4[(b[7] >> 4) & 0xF] * get_scale(base_elem + 14);
    r3.w = c_nf4_v4[ b[7]       & 0xF] * get_scale(base_elem + 15);

    // ── 4. 4× 128-bit 向量写出 ──
    float4* out4 = reinterpret_cast<float4*>(out) + (base_elem >> 2);
    out4[0] = r0;
    out4[1] = r1;
    out4[2] = r2;
    out4[3] = r3;
}