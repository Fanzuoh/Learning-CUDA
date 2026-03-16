#include "nf4_dequant.cuh"

// ── NF4 码表，存到 constant memory（FP32，计算时用）──
__constant__ float c_nf4[16] = {
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

__global__ void nf4_dequant_kernel_v3(
    const uint8_t*  __restrict__ packed,
    const uint8_t*  __restrict__ absmax_q,
    const __half*   __restrict__ code2,       // FP16
    const __half*   __restrict__ absmax2,     // FP16
    uint32_t*       __restrict__ out,         // 每个 uint32_t 存 2 个 BF16
    float           offset,
    int             N,
    int             blocksize,
    int             blocksize2
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_N = (N + 1) >> 1;   // 向上取整，支持奇数N
    if (tid >= half_N) return;

    // ── 1. 读取 packed byte，解出两个 4-bit 索引 ──
    uint8_t byte = packed[tid];
    int q0 = (byte >> 4) & 0x0F;   // 高4位 → 元素 i0
    int q1 =  byte       & 0x0F;   // 低4位 → 元素 i1

    int i0 = tid * 2;
    int i1 = i0 + 1;

    // ── 2. 计算 scale（FP16 → FP32 计算，精度更稳） ──
    auto get_scale = [&](int elem_idx) -> float {
        int bid = elem_idx / blocksize;
        int gid = bid / blocksize2;
        float aq = __half2float(code2[absmax_q[bid]]);
        float a2 = __half2float(absmax2[gid]);
        return aq * a2 + offset;
    };

    float scale0 = get_scale(i0);
    float scale1 = (i1 < N) ? get_scale(i1) : 0.0f;

    // ── 3. 计算 BF16 结果 ──
    __nv_bfloat16 v0 = __float2bfloat16(c_nf4[q0] * scale0);
    __nv_bfloat16 v1 = __float2bfloat16(c_nf4[q1] * scale1);

    // ── 4. 打包成 uint32_t，一次写入（向量化写） ──
    // 内存布局：低16位 = v0（i0），高16位 = v1（i1）
    if (i1 < N) {
        // 正常情况：两个元素都有效
        uint16_t b0, b1;
        memcpy(&b0, &v0, sizeof(uint16_t));
        memcpy(&b1, &v1, sizeof(uint16_t));
        out[tid] = (uint32_t)b0 | ((uint32_t)b1 << 16);
    } else {
        // 边界：N 是奇数，最后只写 1 个 BF16
        uint16_t b0;
        memcpy(&b0, &v0, sizeof(uint16_t));
        reinterpret_cast<uint16_t*>(out)[i0] = b0;
    }
}
__global__ void nf4_dequant_kernel(
    const uint8_t*  __restrict__ packed,
    const uint8_t*  __restrict__ absmax_q,
    const float*    __restrict__ code2,
    const float*    __restrict__ absmax2,
    float*          __restrict__ out,
    float           offset,
    int             N,
    int             blocksize,
    int             blocksize2
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_N = (N + 1) >> 1;
    if (tid >= half_N) return;

    uint8_t byte = packed[tid];
    int q0 = (byte >> 4) & 0x0F;
    int q1 =  byte       & 0x0F;

    int i0 = tid * 2;
    int i1 = i0 + 1;

    auto get_scale = [&](int elem_idx) -> float {
        int bid = elem_idx / blocksize;
        int gid = bid / blocksize2;
        float aq = code2[absmax_q[bid]];
        float a2 = absmax2[gid];
        return aq * a2 + offset;
    };

    float scale0 = get_scale(i0);
    out[i0] = c_nf4[q0] * scale0;

    if (i1 < N) {
        float scale1 = get_scale(i1);
        out[i1] = c_nf4[q1] * scale1;
    }
}
