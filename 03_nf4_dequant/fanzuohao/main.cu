#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "nf4_dequant.cuh"
#include "nf4_dequant_v2.cuh"
#include "nf4_dequant_v4.cuh"
#include "nf4_dequant_v5.cuh"
#include "nf4_dequant_v7.cuh"
#include "nf4_dequant_v8.cuh"
#include "nf4_gemv.cuh"

// ─────────────────────────────────────────
//  二进制文件 Header（兼容旧格式）
//  旧格式: magic/version/N/num_blocks/num_groups/blocksize/blocksize2/offset
//  新格式: 同上，但 N = num_rows * num_cols
// ─────────────────────────────────────────
struct Header {
    uint32_t magic, version;
    uint64_t N;                  // 总元素数 = num_rows * num_cols
    uint32_t num_blocks, num_groups, blocksize, blocksize2;
    float    offset;
    uint32_t _pad;
};

// ─────────────────────────────────────────
//  文本参数文件结构
// ─────────────────────────────────────────
struct Params {
    int  blocksize;              // 必须与 Header 一致
    char compute_type[16];       // "bf16" 或 "fp16"
    char target_gpu[32];         // "T4" / "A100" 等（影响 grid 策略）
    char output_path[256];       // 输出文件路径
    int  num_rows;               // 可选：矩阵行数
    int  num_cols;               // 可选：矩阵列数
};

#define CHECK_CUDA(call) do {                                        \
    cudaError_t e = (call);                                          \
    if (e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d  %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(e));          \
        exit(1);                                                     \
    }                                                                \
} while(0)

// ─────────────────────────────────────────
//  工具函数
// ─────────────────────────────────────────
static void* read_segment(FILE* fp, long offset, size_t bytes) {
    void* buf = malloc(bytes);
    fseek(fp, offset, SEEK_SET);
    if (fread(buf, 1, bytes, fp) != bytes) {
        fprintf(stderr, "fread fail at offset %ld, bytes %zu\n", offset, bytes);
        exit(1);
    }
    return buf;
}

static float bf16_to_float(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, sizeof(f));
    return f;
}

static float half_to_float_host(__half h) { return __half2float(h); }
static __half float_to_half(float f)       { return __float2half(f); }

// ─────────────────────────────────────────
//  【新增】解析文本参数文件
//  格式示例：
//    blocksize    = 64
//    compute_type = "bf16"
//    target_gpu   = "T4"
//    output_path  = "output.bin"
//    num_rows     = 4096
//    num_cols     = 4096
// ─────────────────────────────────────────
static Params parse_params(const char* path) {
    Params p;
    // 默认值
    p.blocksize  = 64;
    p.num_rows   = -1;
    p.num_cols   = -1;
    strncpy(p.compute_type, "fp16",       sizeof(p.compute_type));
    strncpy(p.target_gpu,   "A100",       sizeof(p.target_gpu));
    strncpy(p.output_path,  "output.bin", sizeof(p.output_path));

    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[warn] 参数文件 %s 不存在，使用默认值\n", path);
        return p;
    }

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        // 跳过注释行
        if (line[0] == '#' || line[0] == '\n') continue;

        // 去掉行尾换行
        line[strcspn(line, "\r\n")] = '\0';

        int   ival; char sval[256];
        if (sscanf(line, "blocksize = %d",                &ival)  == 1)
            p.blocksize = ival;
        else if (sscanf(line, "num_rows = %d",            &ival)  == 1)
            p.num_rows = ival;
        else if (sscanf(line, "num_cols = %d",            &ival)  == 1)
            p.num_cols = ival;
        else if (sscanf(line, "compute_type = \"%255[^\"]\"", sval) == 1)
            strncpy(p.compute_type, sval, sizeof(p.compute_type));
        else if (sscanf(line, "target_gpu = \"%255[^\"]\"",   sval) == 1)
            strncpy(p.target_gpu, sval, sizeof(p.target_gpu));
        else if (sscanf(line, "output_path = \"%255[^\"]\"",  sval) == 1)
            strncpy(p.output_path, sval, sizeof(p.output_path));
    }
    fclose(fp);
    return p;
}

// ─────────────────────────────────────────
//  【新增】计算有效内存带宽
//  读: packed(N/2) + absmax_q(num_blocks) + code2(256*4) + absmax2(num_groups*4)
//  写: out(N * elem_bytes)
// ─────────────────────────────────────────
static float calc_bandwidth_GBs(uint64_t N, uint32_t num_blocks,
                                 uint32_t num_groups, int elem_bytes,
                                 float ms_per_iter) {
    double bytes = (double)(N / 2)                   // packed
                 + (double)num_blocks                 // absmax_q
                 + 256.0 * sizeof(float)              // code2
                 + (double)num_groups * sizeof(float) // absmax2
                 + (double)N * elem_bytes;            // output
    return (float)(bytes / (ms_per_iter * 1e-3) / 1e9);
}

// ─────────────────────────────────────────
//  【新增】写输出二进制文件
// ─────────────────────────────────────────
static void write_output(const char* path, const void* data,
                          size_t elem_size, size_t count,
                          const char* compute_type,
                          uint64_t N, int num_rows, int num_cols) {
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "[error] 无法写出文件 %s\n", path); return; }

    // 写简单文件头：magic + dtype + rows + cols
    uint32_t magic = 0x4F555446; // "OUTF"
    uint8_t  dtype = (strncmp(compute_type, "bf16", 4) == 0) ? 1 : 0; // 0=fp16, 1=bf16
    int32_t  rows  = (num_rows > 0) ? num_rows : (int)(N);
    int32_t  cols  = (num_cols > 0) ? num_cols : 1;

    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&dtype, sizeof(dtype), 1, fp);
    fwrite(&rows,  sizeof(rows),  1, fp);
    fwrite(&cols,  sizeof(cols),  1, fp);
    fwrite(data,   elem_size,     count, fp);
    fclose(fp);
    printf("  → 已写出 %s  [%d×%d, %s, %.2f MB]\n",
           path, rows, cols, compute_type,
           (double)(elem_size * count) / 1024.0 / 1024.0);
}

// ─────────────────────────────────────────
//  【新增】根据 target_gpu 调整 block size
// ─────────────────────────────────────────
static int get_block_size(const char* target_gpu) {
    // T4: SM75, 较小 L1，128 线程更优
    // A100/V100: SM80/SM70, 256 线程更优
    if (strncmp(target_gpu, "T4", 2) == 0) return 128;
    return 256;
}

// ─────────────────────────────────────────
//  main
// ─────────────────────────────────────────
int main(int argc, char** argv)
{
    const char* bin_path    = argc > 1 ? argv[1] : "nf4_dq_test.bin";
    const char* params_path = argc > 2 ? argv[2] : "params.txt";

    // ══ 1. 解析文本参数文件 ══════════════
    Params params = parse_params(params_path);
    printf("══ 参数文件 (%s) ══\n", params_path);
    printf("  blocksize    = %d\n",  params.blocksize);
    printf("  compute_type = %s\n",  params.compute_type);
    printf("  target_gpu   = %s\n",  params.target_gpu);
    printf("  output_path  = %s\n",  params.output_path);
    if (params.num_rows > 0)
        printf("  num_rows     = %d\n",  params.num_rows);
    if (params.num_cols > 0)
        printf("  num_cols     = %d\n",  params.num_cols);

    const bool use_bf16 = (strncmp(params.compute_type, "bf16", 4) == 0);
    const int  BLOCK    = get_block_size(params.target_gpu);
    printf("  → 输出类型: %s  线程块大小: %d\n\n",
           use_bf16 ? "BF16" : "FP16", BLOCK);

    // ══ 2. 读二进制文件 ══════════════════
    FILE* fp = fopen(bin_path, "rb");
    if (!fp) { perror("open"); return 1; }

    Header hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "读 Header 失败\n"); return 1;
    }
    printf("══ 数据文件 (%s) ══\n", bin_path);
    printf("  magic=0x%08X  N=%llu  num_blocks=%u  num_groups=%u\n",
           hdr.magic, (unsigned long long)hdr.N,
           hdr.num_blocks, hdr.num_groups);
    printf("  blocksize=%u  blocksize2=%u  offset=%.8f\n",
           hdr.blocksize, hdr.blocksize2, hdr.offset);

    // 参数文件与数据文件 blocksize 一致性校验
    if (params.blocksize != (int)hdr.blocksize) {
        fprintf(stderr, "[warn] params.blocksize=%d 与 hdr.blocksize=%u 不一致，以 hdr 为准\n",
                params.blocksize, hdr.blocksize);
    }

    // 推断 rows/cols
    int num_rows = params.num_rows;
    int num_cols = params.num_cols;
    if (num_rows <= 0 || num_cols <= 0) {
        // 无法从 Header 直接获取，尝试用参数文件，否则默认方阵
        uint64_t sq = (uint64_t)sqrt((double)hdr.N);
        if (sq * sq == hdr.N) { num_rows = num_cols = (int)sq; }
        else                  { num_rows = 1; num_cols = (int)hdr.N; }
        printf("  [推断] num_rows=%d  num_cols=%d\n", num_rows, num_cols);
    }

    // log2 参数
    int log2_bs  = 0; { uint32_t v = hdr.blocksize;  while (v >>= 1) log2_bs++;  }
    int log2_bs2 = 0; { uint32_t v = hdr.blocksize2; while (v >>= 1) log2_bs2++; }
    printf("  log2_blocksize=%d  log2_blocksize2=%d\n\n", log2_bs, log2_bs2);

    // 文件偏移
    long off_packed   = sizeof(hdr);
    long off_absmax_q = off_packed   + (long)(hdr.N / 2);
    long off_code2    = off_absmax_q + (long)hdr.num_blocks;
    long off_absmax2  = off_code2    + 256 * sizeof(float);
    long off_gt       = off_absmax2  + (long)hdr.num_groups * sizeof(float);

    uint8_t* h_packed     = (uint8_t*)read_segment(fp, off_packed,   hdr.N / 2);
    uint8_t* h_absmax_q   = (uint8_t*)read_segment(fp, off_absmax_q, hdr.num_blocks);
    float*   h_code2_f32  = (float*)  read_segment(fp, off_code2,    256 * sizeof(float));
    float*   h_absmax2_f32= (float*)  read_segment(fp, off_absmax2,  hdr.num_groups * sizeof(float));
    float*   h_gt         = (float*)  read_segment(fp, off_gt,       hdr.N * sizeof(float));
    fclose(fp);

    // FP32 → FP16（供 v3/v8 使用）
    __half* h_code2   = (__half*)malloc(256 * sizeof(__half));
    __half* h_absmax2 = (__half*)malloc(hdr.num_groups * sizeof(__half));
    for (int i = 0; i < 256; i++)
        h_code2[i] = float_to_half(h_code2_f32[i]);
    for (uint32_t i = 0; i < hdr.num_groups; i++)
        h_absmax2[i] = float_to_half(h_absmax2_f32[i]);

    printf("文件读取完成，code2/absmax2 已转换为 FP16\n\n");

    // ══ 3. Device 分配 ═══════════════════
    uint8_t  *d_packed, *d_absmax_q;
    __half   *d_code2, *d_absmax2;
    float    *d_code2_f32, *d_absmax2_f32;
    uint32_t *d_out_v3;
    __half   *d_out_half;
    float    *d_out_f32;

    CHECK_CUDA(cudaMalloc(&d_packed,      hdr.N / 2));
    CHECK_CUDA(cudaMalloc(&d_absmax_q,    hdr.num_blocks));
    CHECK_CUDA(cudaMalloc(&d_code2,       256 * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_absmax2,     hdr.num_groups * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_code2_f32,   256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_absmax2_f32, hdr.num_groups * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_v3,      ((hdr.N + 1) / 2) * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_out_half,    hdr.N * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out_f32,     hdr.N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_packed,       h_packed,       hdr.N / 2,                       cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q,     h_absmax_q,     hdr.num_blocks,                  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2,        h_code2,        256 * sizeof(__half),             cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2,      h_absmax2,      hdr.num_groups * sizeof(__half),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2_f32,    h_code2_f32,    256 * sizeof(float),              cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2_f32,  h_absmax2_f32,  hdr.num_groups * sizeof(float),   cudaMemcpyHostToDevice));

    float*    h_out_f32  = (float*)   malloc(hdr.N * sizeof(float));
    __half*   h_out_half = (__half*)  malloc(hdr.N * sizeof(__half));
    uint16_t* h_out_bf16 = (uint16_t*)malloc(hdr.N * sizeof(uint16_t));

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    const int RUNS = 10;

    // ══ 4. 各版本 Kernel 测试 ════════════

#define BENCH_HEADER(name) \
    printf("══════════════════════════════\n  %s\n══════════════════════════════\n", name)

#define BENCH_RESULT(ms_total, mae, max_err, thresh, bw, spd) \
    printf("  平均耗时 : %.3f ms\n  有效带宽 : %.1f GB/s\n"  \
           "  MAE : %.2e  MaxErr : %.2e  %s  加速比 %.2fx\n", \
           (ms_total)/RUNS, bw, mae, max_err,                  \
           (mae) < (thresh) ? "✅ PASS" : "❌ FAIL", spd)

    // ── v1 baseline ──────────────────────
    BENCH_HEADER("v1 kernel（baseline，FP32输出）");
    {
        // 【修复】grid 向上取整，覆盖 N%2 != 0 的情况
        int grid = ((int)(hdr.N + 1) / 2 + BLOCK - 1) / BLOCK;
        nf4_dequant_kernel<<<grid, BLOCK>>>(
            d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
            hdr.offset, (int)hdr.N, (int)hdr.blocksize, (int)hdr.blocksize2);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(t0));
        for (int r = 0; r < RUNS; r++)
            nf4_dequant_kernel<<<grid, BLOCK>>>(
                d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                hdr.offset, (int)hdr.N, (int)hdr.blocksize, (int)hdr.blocksize2);
        CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
        float ms_v1 = 0; CHECK_CUDA(cudaEventElapsedTime(&ms_v1, t0, t1));
        CHECK_CUDA(cudaMemcpy(h_out_f32, d_out_f32, hdr.N * sizeof(float), cudaMemcpyDeviceToHost));

        double mae = 0, max_err = 0;
        for (size_t i = 0; i < hdr.N; i++) {
            double d = fabs((double)h_out_f32[i] - (double)h_gt[i]);
            mae += d; if (d > max_err) max_err = d;
        }
        mae /= hdr.N;
        float bw = calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 4, ms_v1/RUNS);
        printf("  平均耗时 : %.3f ms\n  有效带宽 : %.1f GB/s\n"
               "  MAE : %.2e  MaxErr : %.2e  %s\n",
               ms_v1/RUNS, bw, mae, max_err, mae < 1e-5 ? "✅ PASS" : "❌ FAIL");

        // 保存 v1 结果供 GEMV 使用，同时记录 baseline 时间
        float* h_dq = (float*)malloc(hdr.N * sizeof(float));
        memcpy(h_dq, h_out_f32, hdr.N * sizeof(float));
        float ms_v1_ref = ms_v1; // 供后续加速比计算

        // ── v2 ──────────────────────────
        BENCH_HEADER("v2 kernel（FP32输出）");
        {
            int grid2 = ((int)(hdr.N / 8) + BLOCK - 1) / BLOCK;
            int smem2 = (256 + (int)hdr.num_groups) * sizeof(float);
            nf4_dequant_v2_kernel<<<grid2, BLOCK, smem2>>>(
                d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                hdr.offset, (int)hdr.N, (int)hdr.blocksize, (int)hdr.blocksize2);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t0));
            for (int r = 0; r < RUNS; r++)
                nf4_dequant_v2_kernel<<<grid2, BLOCK, smem2>>>(
                    d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                    hdr.offset, (int)hdr.N, (int)hdr.blocksize, (int)hdr.blocksize2);
            CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
            float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
            CHECK_CUDA(cudaMemcpy(h_out_f32, d_out_f32, hdr.N * sizeof(float), cudaMemcpyDeviceToHost));
            double mae2 = 0, me2 = 0;
            for (size_t i = 0; i < hdr.N; i++) {
                double d = fabs((double)h_out_f32[i] - (double)h_gt[i]);
                mae2 += d; if (d > me2) me2 = d;
            }
            mae2 /= hdr.N;
            float bw2 = calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 4, ms/RUNS);
            BENCH_RESULT(ms, mae2, me2, 1e-5, bw2, (ms_v1_ref/RUNS)/(ms/RUNS));
        }

        // ── v3 BF16 ──────────────────────
        BENCH_HEADER("v3 kernel（BF16输出 + 向量化写入）");
        {
            int grid3 = ((int)((hdr.N + 1) / 2) + BLOCK - 1) / BLOCK;
            nf4_dequant_kernel_v3<<<grid3, BLOCK>>>(
                d_packed, d_absmax_q, d_code2, d_absmax2, d_out_v3,
                hdr.offset, (int)hdr.N, (int)hdr.blocksize, (int)hdr.blocksize2);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t0));
            for (int r = 0; r < RUNS; r++)
                nf4_dequant_kernel_v3<<<grid3, BLOCK>>>(
                    d_packed, d_absmax_q, d_code2, d_absmax2, d_out_v3,
                    hdr.offset, (int)hdr.N, (int)hdr.blocksize, (int)hdr.blocksize2);
            CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
            float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
            CHECK_CUDA(cudaMemcpy(h_out_bf16, d_out_v3,
                                  hdr.N * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            double mae3 = 0, me3 = 0;
            for (size_t i = 0; i < hdr.N; i++) {
                float val = bf16_to_float(h_out_bf16[i]);
                double d  = fabs((double)val - (double)h_gt[i]);
                mae3 += d; if (d > me3) me3 = d;
            }
            mae3 /= hdr.N;
            float bw3 = calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 2, ms/RUNS);
            BENCH_RESULT(ms, mae3, me3, 1e-2, bw3, (ms_v1_ref/RUNS)/(ms/RUNS));
        }

        // ── v4 ──────────────────────────
        BENCH_HEADER("v4 kernel（每线程16元素，FP32输出）");
        {
            // 【修复】N 不一定是 16 的倍数，向上取整
            int grid4 = ((int)(hdr.N + 15) / 16 + BLOCK - 1) / BLOCK;
            nf4_dequant_v4_kernel<<<grid4, BLOCK>>>(
                d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                hdr.offset, (int)hdr.N, log2_bs, log2_bs2);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t0));
            for (int r = 0; r < RUNS; r++)
                nf4_dequant_v4_kernel<<<grid4, BLOCK>>>(
                    d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                    hdr.offset, (int)hdr.N, log2_bs, log2_bs2);
            CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
            float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
            CHECK_CUDA(cudaMemcpy(h_out_f32, d_out_f32, hdr.N * sizeof(float), cudaMemcpyDeviceToHost));
            double mae4 = 0, me4 = 0;
            for (size_t i = 0; i < hdr.N; i++) {
                double d = fabs((double)h_out_f32[i] - (double)h_gt[i]);
                mae4 += d; if (d > me4) me4 = d;
            }
            mae4 /= hdr.N;
            float bw4 = calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 4, ms/RUNS);
            BENCH_RESULT(ms, mae4, me4, 1e-5, bw4, (ms_v1_ref/RUNS)/(ms/RUNS));
        }

        // ── v5 ──────────────────────────
        BENCH_HEADER("v5 kernel（shared mem scale，FP32输出）");
        {
            int grid5 = ((int)(hdr.N + 15) / 16 + BLOCK - 1) / BLOCK;
            int smem5 = (BLOCK * 16 / (int)hdr.blocksize) * (int)sizeof(float);
            nf4_dequant_v5_kernel<<<grid5, BLOCK, smem5>>>(
                d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                hdr.offset, (int)hdr.N, log2_bs, log2_bs2);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t0));
            for (int r = 0; r < RUNS; r++)
                nf4_dequant_v5_kernel<<<grid5, BLOCK, smem5>>>(
                    d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                    hdr.offset, (int)hdr.N, log2_bs, log2_bs2);
            CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
            float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
            CHECK_CUDA(cudaMemcpy(h_out_f32, d_out_f32, hdr.N * sizeof(float), cudaMemcpyDeviceToHost));
            double mae5 = 0, me5 = 0;
            for (size_t i = 0; i < hdr.N; i++) {
                double d = fabs((double)h_out_f32[i] - (double)h_gt[i]);
                mae5 += d; if (d > me5) me5 = d;
            }
            mae5 /= hdr.N;
            float bw5 = calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 4, ms/RUNS);
            BENCH_RESULT(ms, mae5, me5, 1e-5, bw5, (ms_v1_ref/RUNS)/(ms/RUNS));
        }

        // ── v7 ──────────────────────────
        BENCH_HEADER("v7 kernel（FP32输出）");
        {
            int grid7 = ((int)(hdr.N + 15) / 16 + BLOCK - 1) / BLOCK;
            nf4_dequant_v7_kernel<<<grid7, BLOCK>>>(
                d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                hdr.offset, (int)hdr.N, log2_bs, log2_bs2);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t0));
            for (int r = 0; r < RUNS; r++)
                nf4_dequant_v7_kernel<<<grid7, BLOCK>>>(
                    d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_f32,
                    hdr.offset, (int)hdr.N, log2_bs, log2_bs2);
            CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
            float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
            CHECK_CUDA(cudaMemcpy(h_out_f32, d_out_f32, hdr.N * sizeof(float), cudaMemcpyDeviceToHost));
            double mae7 = 0, me7 = 0;
            for (size_t i = 0; i < hdr.N; i++) {
                double d = fabs((double)h_out_f32[i] - (double)h_gt[i]);
                mae7 += d; if (d > me7) me7 = d;
            }
            mae7 /= hdr.N;
            float bw7 = calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 4, ms/RUNS);
            BENCH_RESULT(ms, mae7, me7, 1e-5, bw7, (ms_v1_ref/RUNS)/(ms/RUNS));
        }

        // ── v8 FP16/BF16 输出（最优版本）──
        BENCH_HEADER("v8 kernel（__half 输出，最优版本）");
        float ms_v8_best = 0;
        {
            int grid8 = ((int)(hdr.N + 15) / 16 + BLOCK - 1) / BLOCK;
            nf4_dequant_v8_kernel<<<grid8, BLOCK>>>(
                d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_half,
                hdr.offset, (int)hdr.N, log2_bs, log2_bs2);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t0));
            for (int r = 0; r < RUNS; r++)
                nf4_dequant_v8_kernel<<<grid8, BLOCK>>>(
                    d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_out_half,
                    hdr.offset, (int)hdr.N, log2_bs, log2_bs2);
            CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
            float ms = 0; CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
            ms_v8_best = ms;
            CHECK_CUDA(cudaMemcpy(h_out_half, d_out_half,
                                  hdr.N * sizeof(__half), cudaMemcpyDeviceToHost));
            double mae8 = 0, me8 = 0;
            for (size_t i = 0; i < hdr.N; i++) {
                float val = half_to_float_host(h_out_half[i]);
                double d  = fabs((double)val - (double)h_gt[i]);
                mae8 += d; if (d > me8) me8 = d;
            }
            mae8 /= hdr.N;
            float bw8 = calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 2, ms/RUNS);
            BENCH_RESULT(ms, mae8, me8, 1e-2, bw8, (ms_v1_ref/RUNS)/(ms/RUNS));
        }

        // ══ 5. 汇总表 ════════════════════
        printf("\n══════════════════════════════════════════════════════════════════\n");
        printf("  %-8s  %8s  %10s  %6s\n", "kernel", "ms/iter", "BW(GB/s)", "speedup");
        // 只打印 v1 和 v8 的汇总（其他已在各自块内打印）
        printf("  %-8s  %8.3f  %10.1f  %6s\n", "v1",
               ms_v1_ref/RUNS,
               calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 4, ms_v1_ref/RUNS),
               "1.00x");
        printf("  %-8s  %8.3f  %10.1f  %6.2fx  ← 最优\n", "v8",
               ms_v8_best/RUNS,
               calc_bandwidth_GBs(hdr.N, hdr.num_blocks, hdr.num_groups, 2, ms_v8_best/RUNS),
               (ms_v1_ref/RUNS)/(ms_v8_best/RUNS));
        printf("══════════════════════════════════════════════════════════════════\n\n");

        // ══ 6. 写输出文件 ════════════════
        // 根据 compute_type 选择写 FP16 还是 BF16
        if (use_bf16) {
            // v3 输出是 BF16，写 h_out_bf16
            write_output(params.output_path,
                         h_out_bf16, sizeof(uint16_t), hdr.N,
                         "bf16", hdr.N, num_rows, num_cols);
        } else {
            // v8 输出是 FP16，写 h_out_half
            write_output(params.output_path,
                         h_out_half, sizeof(__half), hdr.N,
                         "fp16", hdr.N, num_rows, num_cols);
        }

        // ══ 7. GEMV fused kernel ═════════
        printf("══════════════════════════════\n  GEMV fused kernel\n══════════════════════════════\n");
        const int GM = num_rows, GK = num_cols;
        if ((size_t)GM * GK != hdr.N) {
            printf("  [skip] N=%llu 与 %d×%d 不匹配\n",
                   (unsigned long long)hdr.N, GM, GK);
        } else {
            float* h_x = (float*)malloc(GK * sizeof(float));
            srand(42);
            for (int i = 0; i < GK; i++)
                h_x[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;

            // CPU 参考
            float* h_y_ref = (float*)malloc(GM * sizeof(float));
            for (int i = 0; i < GM; i++) {
                float acc = 0.f;
                for (int j = 0; j < GK; j++)
                    acc += h_dq[(size_t)i * GK + j] * h_x[j];
                h_y_ref[i] = acc;
            }

            float *d_x, *d_y;
            CHECK_CUDA(cudaMalloc(&d_x, GK * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_y, GM * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_x, h_x, GK * sizeof(float), cudaMemcpyHostToDevice));

            const int WARPS_PB = 8, BLOCK_GEMV = WARPS_PB * 32;
            int grid_gemv = (GM + WARPS_PB - 1) / WARPS_PB;
            int smem_gemv = (256 + GK) * sizeof(float);

            nf4_gemv_kernel<<<grid_gemv, BLOCK_GEMV, smem_gemv>>>(
                d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_x, d_y,
                hdr.offset, GM, GK, (int)hdr.blocksize, (int)hdr.blocksize2);
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t0));
            for (int r = 0; r < RUNS; r++)
                nf4_gemv_kernel<<<grid_gemv, BLOCK_GEMV, smem_gemv>>>(
                    d_packed, d_absmax_q, d_code2_f32, d_absmax2_f32, d_x, d_y,
                    hdr.offset, GM, GK, (int)hdr.blocksize, (int)hdr.blocksize2);
            CHECK_CUDA(cudaEventRecord(t1)); CHECK_CUDA(cudaEventSynchronize(t1));
            float ms_gemv = 0; CHECK_CUDA(cudaEventElapsedTime(&ms_gemv, t0, t1));

            float* h_y = (float*)malloc(GM * sizeof(float));
            CHECK_CUDA(cudaMemcpy(h_y, d_y, GM * sizeof(float), cudaMemcpyDeviceToHost));

            double mae_g = 0, me_g = 0;
            for (int i = 0; i < GM; i++) {
                double d = fabs((double)h_y[i] - (double)h_y_ref[i]);
                mae_g += d; if (d > me_g) me_g = d;
            }
            mae_g /= GM;
            printf("  M=%d  K=%d\n  平均耗时 : %.3f ms\n"
                   "  MAE : %.2e  MaxErr : %.2e  %s\n",
                   GM, GK, ms_gemv/RUNS, mae_g, me_g,
                   mae_g < 1e-1 ? "✅ PASS" : "❌ FAIL");

            cudaFree(d_x); cudaFree(d_y);
            free(h_x); free(h_y); free(h_y_ref);
        }

        free(h_dq);
    } // end v1 scope

    // ══ 8. 清理 ══════════════════════════
    cudaFree(d_packed);    cudaFree(d_absmax_q);
    cudaFree(d_code2);     cudaFree(d_absmax2);
    cudaFree(d_code2_f32); cudaFree(d_absmax2_f32);
    cudaFree(d_out_v3);    cudaFree(d_out_half);
    cudaFree(d_out_f32);
    free(h_packed);    free(h_absmax_q);
    free(h_code2);     free(h_absmax2);
    free(h_code2_f32); free(h_absmax2_f32);
    free(h_gt);        free(h_out_f32);
    free(h_out_half);  free(h_out_bf16);
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    return 0;
}