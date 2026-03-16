// /home/fanzuohao/kernel_test/nf4_dequant_v2.cu
#include "nf4_dequant_v2.cuh"

__constant__ float c_nf4_v2[16] = {
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

__global__ void nf4_dequant_v2_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ absmax_q,
    const float*   __restrict__ code2,
    const float*   __restrict__ absmax2,
    float*         __restrict__ out,
    float          offset,
    int            N,
    int            blocksize,
    int            blocksize2
)
{
    // ── shared memory ──
    // s_code2  : [256]  floats = 1024 bytes
    // s_absmax2: [num_groups] floats，num_groups=1024 → 4096 bytes
    extern __shared__ float smem[];
    float* s_code2   = smem;
    float* s_absmax2 = smem + 256;

    // 协作加载 code2
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        s_code2[i] = __ldg(&code2[i]);

    // 协作加载 absmax2
    int num_groups = N / blocksize / blocksize2;
    for (int i = threadIdx.x; i < num_groups; i += blockDim.x)
        s_absmax2[i] = __ldg(&absmax2[i]);

    __syncthreads();

    // ── 每线程处理 4 个 packed byte = 8 个元素 ──
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    int base_byte = tid * 4;          // packed 起始下标
    int base_elem = base_byte * 2;    // 元素起始下标（= tid * 8）

    if (base_elem >= N) return;

    // scale 计算：blocksize=64 → >>6，blocksize2=256 → >>8
    auto get_scale = [&](int elem_idx) -> float {
        int bid = elem_idx >> 6;
        int gid = bid      >> 8;
        return s_code2[__ldg(&absmax_q[bid])] * s_absmax2[gid] + offset;
    };

    // ── 展开 4 个 byte，计算 8 个输出值 ──
    float4 v0, v1;

    {
        uint8_t b = __ldg(&packed[base_byte + 0]);
        v0.x = c_nf4_v2[(b >> 4) & 0xF] * get_scale(base_elem + 0);
        v0.y = c_nf4_v2[ b       & 0xF] * get_scale(base_elem + 1);
    }
    {
        uint8_t b = __ldg(&packed[base_byte + 1]);
        v0.z = c_nf4_v2[(b >> 4) & 0xF] * get_scale(base_elem + 2);
        v0.w = c_nf4_v2[ b       & 0xF] * get_scale(base_elem + 3);
    }
    {
        uint8_t b = __ldg(&packed[base_byte + 2]);
        v1.x = c_nf4_v2[(b >> 4) & 0xF] * get_scale(base_elem + 4);
        v1.y = c_nf4_v2[ b       & 0xF] * get_scale(base_elem + 5);
    }
    {
        uint8_t b = __ldg(&packed[base_byte + 3]);
        v1.z = c_nf4_v2[(b >> 4) & 0xF] * get_scale(base_elem + 6);
        v1.w = c_nf4_v2[ b       & 0xF] * get_scale(base_elem + 7);
    }

    // ── float4 向量写出（128-bit store）──
    // base_elem 步进 8，base_elem/4 步进 2，对应连续两个 float4
    reinterpret_cast<float4*>(out)[base_elem / 4 + 0] = v0;
    reinterpret_cast<float4*>(out)[base_elem / 4 + 1] = v1;
}