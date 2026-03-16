#include "nf4_dequant_v5.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// ── NF4 码表，constant memory，16个float，warp内广播命中 ──
__constant__ float c_nf4_v5[16] = {
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
//  v5 kernel：预计算 scale 到 shared memory，消除依赖链
//
//  blocksize=64, blocksize2=256, block线程数=256
//  每个 thread block 处理的元素数 = 256 threads × 16 elem = 4096 elem
//  对应的 absmax_q block 数      = 4096 / 64 = 64 个
//  对应的 absmax2 group 数       = 64 / 256 = 0.25 → 最多 1 个 group
//
//  shared memory 布局：
//    smem_scale[64]  : 本 thread block 负责的 64 个 scale 值
//    (code2/absmax2 通过 __ldg 读，不放 smem，避免占用过多 smem)
//
//  流程：
//    Phase 1: 256 线程协作，每线程算 64/256 = 0.25 → 前64线程各算1个scale
//    Phase 2: __syncthreads()
//    Phase 3: 每线程用 smem_scale 做 dequant，无 global mem 依赖
// ────────────────────────────────────────────────────────────────
__global__ void nf4_dequant_v5_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ code2,
    const float*   __restrict__ absmax2,
    float*         __restrict__ out,
    float          offset,
    int            N,
    int            log2_blocksize,    // 64  → 6
    int            log2_blocksize2    // 256 → 8（absmax_q block 数的 log2）
)
{
    // ── 每个 thread block 负责的起始元素 ──
    // 每 block 处理 blockDim.x * 16 个元素
    int block_elem_start = blockIdx.x * (blockDim.x << 4);  // blockIdx.x * 4096

    // ── shared memory：存本 block 负责的所有 scale ──
    // 本 block 覆盖的 absmax_q block 数 = (blockDim.x * 16) / blocksize
    //   = (256 * 16) / 64 = 64
    extern __shared__ float smem_scale[];   // [64]

    // ── Phase 1：前 64 个线程各预计算 1 个 scale ──
    // absmax_q block 起始索引
    int absmax_block_start = block_elem_start >> log2_blocksize;  // / 64

    if (threadIdx.x < blockDim.x) {   // 实际只需 64 个，但用 blockDim.x 保证灵活
        // 每线程算几个 scale
        int scales_per_block = (blockDim.x << 4) >> log2_blocksize;  // 64
        int scales_per_thread = (scales_per_block + blockDim.x - 1) / blockDim.x;
        // scales_per_thread = (64 + 255) / 256 = 1（向上取整）

        for (int s = 0; s < scales_per_thread; s++) {
            int local_bid = threadIdx.x * scales_per_thread + s;
            if (local_bid < scales_per_block) {
                int global_bid = absmax_block_start + local_bid;
                int global_gid = global_bid >> log2_blocksize2;
                smem_scale[local_bid] =
                    __ldg(&code2[__ldg(&absmax_q[global_bid])])
                    * __ldg(&absmax2[global_gid])
                    + offset;
            }
        }
    }
    __syncthreads();

    // ── Phase 2：每线程处理 16 个元素 ──
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    int base_elem = tid << 4;
    if (base_elem >= N) return;

    int base_byte = tid << 3;

    // 64-bit 向量读取
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

    // 从 smem 取 scale（无 global mem 依赖，L1 延迟 ~4 cycle）
    // 线程 tid 的 base_elem 对应的 local_bid：
    int local_bid_base = (threadIdx.x << 4) >> log2_blocksize;  // threadIdx.x * 16 / 64

    // 本线程的 16 个元素最多跨 2 个 absmax_q block
    auto get_scale_smem = [&](int local_elem_idx) -> float {
        int local_bid = (local_elem_idx) >> log2_blocksize;
        return smem_scale[local_bid];
    };

    int local_base = threadIdx.x << 4;  // threadIdx.x * 16

    float4 r0, r1, r2, r3;

    r0.x = c_nf4_v5[(b[0] >> 4) & 0xF] * get_scale_smem(local_base +  0);
    r0.y = c_nf4_v5[ b[0]       & 0xF] * get_scale_smem(local_base +  1);
    r0.z = c_nf4_v5[(b[1] >> 4) & 0xF] * get_scale_smem(local_base +  2);
    r0.w = c_nf4_v5[ b[1]       & 0xF] * get_scale_smem(local_base +  3);

    r1.x = c_nf4_v5[(b[2] >> 4) & 0xF] * get_scale_smem(local_base +  4);
    r1.y = c_nf4_v5[ b[2]       & 0xF] * get_scale_smem(local_base +  5);
    r1.z = c_nf4_v5[(b[3] >> 4) & 0xF] * get_scale_smem(local_base +  6);
    r1.w = c_nf4_v5[ b[3]       & 0xF] * get_scale_smem(local_base +  7);

    r2.x = c_nf4_v5[(b[4] >> 4) & 0xF] * get_scale_smem(local_base +  8);
    r2.y = c_nf4_v5[ b[4]       & 0xF] * get_scale_smem(local_base +  9);
    r2.z = c_nf4_v5[(b[5] >> 4) & 0xF] * get_scale_smem(local_base + 10);
    r2.w = c_nf4_v5[ b[5]       & 0xF] * get_scale_smem(local_base + 11);

    r3.x = c_nf4_v5[(b[6] >> 4) & 0xF] * get_scale_smem(local_base + 12);
    r3.y = c_nf4_v5[ b[6]       & 0xF] * get_scale_smem(local_base + 13);
    r3.z = c_nf4_v5[(b[7] >> 4) & 0xF] * get_scale_smem(local_base + 14);
    r3.w = c_nf4_v5[ b[7]       & 0xF] * get_scale_smem(local_base + 15);

    // 128-bit 向量写出
    float4* out4 = reinterpret_cast<float4*>(out) + (base_elem >> 2);
    out4[0] = r0;
    out4[1] = r1;
    out4[2] = r2;
    out4[3] = r3;
}