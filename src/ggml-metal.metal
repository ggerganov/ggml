#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define QK4_0 32
#define QR4_0 2
typedef struct {
    half    d;             // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK4_1 32
typedef struct {
    half d;          // delta
    half m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;

static void dequantize_row_q4_0(device const block_q4_0 * x, device float * y, int k) {
    const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const half d = x[i].d;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

static void dequantize_row_q4_1(device const block_q4_1 * x, device float * y, int k) {
    const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const half d = x[i].d;
        const half m = x[i].m;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = x0*d + m;
            y[i*qk + j + qk/2] = x1*d + m;
        }
    }
}

kernel void kernel_add(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig];
}

kernel void kernel_mul(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig];
}

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_mul_row(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % ne00];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_silu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    float x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

constant float GELU_COEF_A    = 0.044715f;
constant float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

kernel void kernel_gelu(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    float x = src0[tpig];
    dst[tpig] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_soft_max(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        threadgroup float  * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    device const float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;
    device       float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    // parallel max
    buf[tpitg[0]] = -INFINITY;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        buf[tpitg[0]] = MAX(buf[tpitg[0]], psrc0[i00]);
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg[0]/2; i > 0; i /= 2) {
        if (tpitg[0] < i) {
            buf[tpitg[0]] = MAX(buf[tpitg[0]], buf[tpitg[0] + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg[0] == 0) {
        buf[0] = buf[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float max = buf[0];

    // parallel sum
    buf[tpitg[0]] = 0.0f;
    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        buf[tpitg[0]] += exp(psrc0[i00] - max);
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg[0]/2; i > 0; i /= 2) {
        if (tpitg[0] < i) {
            buf[tpitg[0]] += buf[tpitg[0] + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg[0] == 0) {
        buf[0] = buf[0];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sum = buf[0];

    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {
        pdst[i00] = exp(psrc0[i00] - max) / sum;
    }
}

kernel void kernel_diag_mask_inf(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant       int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i02 = tpig[2];
    const int64_t i01 = tpig[1];
    const int64_t i00 = tpig[0];

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_get_rows_f16(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    for (int j = 0; j < ne00; j++) {
        dst[i*nb1 + j] = ((device half *) ((device char *) src0 + r*nb01))[j];
    }
}

kernel void kernel_get_rows_q4_0(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_0(
            (device const block_q4_0 *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q4_1(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_1(
            (device const block_q4_1 *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);
    // MEAN
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += x[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // broadcast
    if (tpitg == 0) {
        sum[0] /= ne00;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean  = sum[0];

    // recenter
    device float * y = dst + tgpig*ne00;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] - mean;
    }

    // VARIANCE
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += y[i00] * y[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // broadcast
    if (tpitg == 0) {
        sum[0] /= ne00;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float variance = sum[0];

    const float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = y[i00] * scale;
    }
}


kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);

    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += x[i00] * x[i00];
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // broadcast
    if (tpitg == 0) {
        sum[0] /= ne00;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean  = sum[0];
    const float scale = 1.0f/sqrt(mean + eps);

    device float * y = dst + tgpig*ne00;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_mul_mat_q4_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {
    const int nb = ne00/QK4_0;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q4_0 * x = (device const block_q4_0 *) src0 + r0*nb;
    device const float      * y = (device const float      *) src1 + r1*ne10;

    const int nth = tptg.x*tptg.y;
    const int ith = tptg.y*tpitg.x + tpitg.y;

    const int ix = tpitg.y/4;           // 0 or 1
    const int iy = tpitg.y - 4*ix;      // 0...3

    const int first = 4 * iy;

    float sumf = 0;

    for (int i = 2*tpitg.x + ix; i < nb; i += 2*tptg.x) {

        const float d = (float)x[i].d;

        device const uint8_t * xl = x[i].qs + first;
        device const float   * yl = y + i * QK4_0 + first;

        float2 acc = {0.0f, 0.0f};

        for (int j = 0; j < 4; ++j) {

            acc[0] += yl[j] * (xl[j] & 0xF) + yl[j+16] * (xl[j] >> 4);
            acc[1] += yl[j] + yl[j+16];

        }

        sumf += d * (acc[0] - 8.f*acc[1]);
    }

    sum[ith] = sumf;

    //
    // Accumulate the sum from all threads in the threadgroup
    //
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%4 == 0) {
        sum[ith] += sum[ith+1] + sum[ith+2] + sum[ith+3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%16 == 0) {
        sum[ith] += sum[ith+4] + sum[ith+8] + sum[ith+12];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith == 0) {
        for (uint i = 16; i < nth; i += 16) sum[0] += sum[i];
        dst[r1*ne0 + r0] = sum[0];
    }
}

kernel void kernel_mul_mat_q4_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {
    const int nb = ne00/QK4_1;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q4_1 * x = (device const block_q4_1 *) src0 + r0*nb;
    device const float      * y = (device const float      *) src1 + r1*ne10;

    const uint nth = tptg.x*tptg.y;
    const uint ith = tptg.y*tpitg.x + tpitg.y;

    const int ix = tpitg.y/4;           // 0 or 1
    const int iy = tpitg.y - 4*ix;      // 0...3

    const int first = 4 * iy;

    float sumf = 0;

    for (int i = 2*tpitg.x + ix; i < nb; i += 2*tptg.x) {

        const float d = (float)x[i].d;
        const float m = (float)x[i].m;

        device const uint8_t * xl = x[i].qs + first;
        device const float   * yl = y + i * QK4_1 + first;

        float2 acc = {0.0f, 0.0f};

        for (int j = 0; j < 4; ++j) {

            acc[0] += yl[j+ 0] * (d * (xl[j] & 0xF) + m);
            acc[1] += yl[j+16] * (d * (xl[j] >>  4) + m);

        }

        sumf += acc[0] + acc[1];
    }

    sum[ith] = sumf;

    //
    // Accumulate the sum from all threads in the threadgroup
    //
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%4 == 0) {
        sum[ith] += sum[ith+1] + sum[ith+2] + sum[ith+3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%16 == 0) {
        sum[ith] += sum[ith+4] + sum[ith+8] + sum[ith+12];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith == 0) {
        for (int i = 16; i < nth; i += 16) sum[0] += sum[i];
        dst[r1*ne0 + r0] = sum[0];
    }
}

kernel void kernel_mul_mat_f16_f32(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        threadgroup float  * sum [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tpig[[thread_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    device const half  * x = (device const half  *) (src0 + r0*nb01 + im*nb02);
    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

    sum[tpitg.x] = 0.0f;

    for (int i = tpitg.x; i < ne00; i += tptg.x) {
        sum[tpitg.x] += (float) x[i] * (float) y[i];
    }

    // accumulate the sum from all threads in the threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tptg.x/2; i > 0; i /= 2) {
        if (tpitg.x < i) {
            sum[tpitg.x] += sum[tpitg.x + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tpitg.x == 0) {
        dst[im*ne1*ne0 + r1*ne0 + r0] = sum[0];
    }
}

kernel void kernel_alibi_f32(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        constant      float & m0,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
    float m_k = pow(m0, i2 + 1);
    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0] + m_k * (i00 - ne00 + 1);
    }
}

kernel void kernel_rope(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        constant       int & n_past,
        constant       int & n_dims,
        constant       int & mode,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i3 = tpig[2];
    const int64_t i2 = tpig[1];
    const int64_t i1 = tpig[0];

    const bool is_neox = mode & 2;
    const float theta_scale = pow(10000.0, -2.0f/n_dims);

    const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);

    float theta = (float)p;

    if (!is_neox) {
        for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cos(theta);
            const float sin_theta = sin(theta);

            theta *= theta_scale;

            device const float * const src = (device float *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       float * dst_data  = (device float *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        }
    } else {
        // TODO: implement
    }
}

kernel void kernel_cpy_f16_f16(
        device const half * src0,
        device       half * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const half * src = (device half *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f16(
        device const float * src0,
        device        half * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f32(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

//============================================ k-quants ======================================================

#define QK_K 256

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half d;           // super-block scale for quantized scales
    half dmin;        // super-block scale for quantized mins
} block_q2_k;
// 84 bytes / block

typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    half d;                    // super-block scale
} block_q3_k;
// 110 bytes / block

typedef struct {
    half d;             // super-block scale for quantized scales
    half dmin;          // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_k;
// 144 bytes / block

typedef struct {
    half d;                      // super-block scale for quantized scales
    half dmin;                   // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64];   // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_k;
// 176 bytes / block

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;                  // super-block scale
} block_q6_k;
// 210 bytes / block

static inline uchar4 get_scale_min_k4(int j, device const uint8_t * q) {
    uchar4 r;
    if (j < 4) {
        r[0] = q[j+0] & 63;
        r[2] = q[j+1] & 63;
        r[1] = q[j+4] & 63;
        r[3] = q[j+5] & 63;
    } else {
        r[0] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        r[2] = (q[j+5] & 0xF) | ((q[j-3] >> 6) << 4);
        r[1] = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
        r[3] = (q[j+5] >>  4) | ((q[j+1] >> 6) << 4);
    }
    return r;
}

//========================================== dequantization =============================

static void dequantize_row_q2_k(device const block_q2_k * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = x[i].d;
        const float min = x[i].dmin;

        device const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }

    }
}

static void dequantize_row_q3_k(device const block_q3_k * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    uint16_t aux[8];
    thread const int8_t * scales = (thread const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = (float)(x[i].d);

        device const uint8_t * q = x[i].qs;
        device const uint8_t * h = x[i].hmask;
        uint8_t m = 1;

        device const uint16_t * a = (device const uint16_t *)x[i].scales;
        aux[0] = (a[0] & kmask2) | (((a[4] >> 0) & kmask1) << 4);
        aux[1] = (a[1] & kmask2) | (((a[5] >> 0) & kmask1) << 4);
        aux[2] = (a[2] & kmask2) | (((a[4] >> 2) & kmask1) << 4);
        aux[3] = (a[3] & kmask2) | (((a[5] >> 2) & kmask1) << 4);
        aux[4] = ((a[0] >> 4) & kmask2) | (((a[4] >> 4) & kmask1) << 4);
        aux[5] = ((a[1] >> 4) & kmask2) | (((a[5] >> 4) & kmask1) << 4);
        aux[6] = ((a[2] >> 4) & kmask2) | (((a[4] >> 6) & kmask1) << 4);
        aux[7] = ((a[3] >> 4) & kmask2) | (((a[5] >> 6) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((h[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((h[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }

}

static void dequantize_row_q4_k(device const block_q4_k * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;


    for (int i = 0; i < nb; i++) {

        const float d = x[i].d;
        const float min = x[i].dmin;

        device const uint8_t * q = x[i].qs;
        device const uint8_t * scales = x[i].scales;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            const uchar4 sc = get_scale_min_k4(is, scales);
            const float d1 = d * sc[0]; const float m1 = min * sc[1];
            const float d2 = d * sc[2]; const float m2 = min * sc[3];
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }

    }
}

static void dequantize_row_q5_k(device const block_q5_k * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

   for (int i = 0; i < nb; i++) {

        const float d = (float)(x[i].d);
        const float min = (float)(x[i].dmin);

        device const uint8_t * ql = x[i].qs;
        device const uint8_t * qh = x[i].qh;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            const uchar4 sc = get_scale_min_k4(is, x[i].scales);
            const float d1 = d * sc[0]; const float m1 = min * sc[1];
            const float d2 = d * sc[2]; const float m2 = min * sc[3];
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }

}

static void dequantize_row_q6_k(device const block_q6_k * x, device float * y, int k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        device const uint8_t * ql = x[i].ql;
        device const uint8_t * qh = x[i].qh;
        device const int8_t  * sc = x[i].scales;

        const float d = x[i].d;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

kernel void kernel_get_rows_q2_k(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q2_k(
            (device const block_q2_k *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q3_k(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q3_k(
            (device const block_q3_k *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q4_k(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q4_k(
            (device const block_q4_k *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q5_k(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q5_k(
            (device const block_q5_k *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

kernel void kernel_get_rows_q6_k(
        device const  void * src0,
        device const   int * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb1,
        uint tpig[[thread_position_in_grid]]) {
    const int i = tpig;
    const int r = ((device int32_t *) src1)[i];

    dequantize_row_q6_k(
            (device const block_q6_k *) ((device char *) src0 + r*nb01),
                       (device float *) ((device char *)  dst + i*nb1), ne00);
}

//====================================== dot products =========================

kernel void kernel_mul_mat_q2_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q2_k * x = (device const block_q2_k *) src0 + r0*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    const int nth = tptg.x*tptg.y;
    const int ith = tptg.y*tpitg.x + tpitg.y;

    const int tid = tpitg.y;    // 0...16
    const int il  = tid/4;      // 0...3
    const int ir  = tid%4;      // 0...3
    const int ip  = il/2;       // 0 or 1
    const int shift1 = 4*(il%2);// 0 or 4
    const int shift2 = shift1+2;// 2 or 6
    const int n   = 8;
    const int is  = 4*il + (n*ir)/16;

    const int y_offset = 64*il + n*ir;
    const int q_offset = 32*ip + n*ir;

    sum[ith] = 0.0f;

    float sumf = 0;
    for (int i = tpitg.x; i < nb; i += tptg.x) {

        device const uint8_t * q = x[i].qs + q_offset;
        device const uint8_t * scales = x[i].scales + is;

        uint8_t d1 = scales[0] & 0xF;
        uint8_t d2 = scales[2] & 0xF;
        uint8_t m1 = scales[0] >>  4;
        uint8_t m2 = scales[2] >>  4;

        device const float   * y = yy + i*QK_K + y_offset;

        //float4 s = {0.f, 0.f, 0.f, 0.f};
        float2 s = {0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < n; ++l) {
            s[0] += y[l+ 0] * ((q[l] >> shift1) & 3);
            s[1] += y[l+32] * ((q[l] >> shift2) & 3);
            smin += y[l+ 0] * m1 + y[l+32] * m2;
        }

        const float dall = (float)x[i].d;
        const float dmin = (float)x[i].dmin;

        sumf += dall * (s[0] * d1 + s[1] * d2) - dmin * smin;

    }
    sum[ith] = sumf;

    //int mask1 = (ith%4 == 0);
    //int mask2 = (ith%16 == 0);

    //threadgroup_barrier(mem_flags::mem_threadgroup);
    //for (int i = 1; i < 4; ++i) sum[ith] += mask1 * sum[ith + i];
    //threadgroup_barrier(mem_flags::mem_threadgroup);
    //for (int i = 4; i < 16; i += 4) sum[ith] += mask2 * sum[ith + i];
    //threadgroup_barrier(mem_flags::mem_threadgroup);
    //if (ith == 0) {
    //    for (int i = 16; i < nth; i += 16) sum[0] += sum[i];
    //    dst[r1*ne0 + r0] = sum[0];
    //}

    //
    // Accumulate the sum from all threads in the threadgroup
    // This version is slightly faster than the commented out one below,
    // which I copy-pasted from ggerganov's q4_0 dot product for metal.
    //
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%4 == 0) {
        for (int i = 1; i < 4; ++i) sum[ith] += sum[ith + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%16 == 0) {
        for (int i = 4; i < 16; i += 4) sum[ith] += sum[ith + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith == 0) {
        for (int i = 16; i < nth; i += 16) sum[0] += sum[i];
        dst[r1*ne0 + r0] = sum[0];
    }
}

kernel void kernel_mul_mat_q3_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const uint8_t m3 = 3;
    const int8_t  m4 = 4;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q3_k * x = (device const block_q3_k *) src0 + r0*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    const int nth = tptg.x*tptg.y;
    const int ith = tptg.y*tpitg.x + tpitg.y;

    const int tid = tpitg.y;        // expecting 16
    const int ip  = tid/8;          // 0 or 1
    const int il  = tid/2 - 4*ip;   // 0...3
    const int ir  = tid%2;
    const int n   = 8;
    const int l0  = n*ir;

    const uint8_t m = 1 << (4*ip + il);

    const int shift = 2*il;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + 2*(il/2);
    const int ik = 4 + (il%2);

    const int q_offset = 32*ip + l0;
    const int y_offset = 128*ip + 32*il + l0;

    //float sumf = 0;
    float sumf1 = 0, sumf2 = 0;
    for (int i = tpitg.x; i < nb; i += tptg.x) {

        const float d_all = (float)(x[i].d);

        device const uint8_t * q = x[i].qs + q_offset;
        device const uint8_t * h = x[i].hmask + l0;
        device const float   * y = yy + i * QK_K + y_offset;

        device const uint16_t * a = (device const uint16_t *)x[i].scales;
        const char2 scales = as_type<char2>((uint16_t)(((a[il] >> s_shift1) & kmask2) | (((a[ik] >> s_shift2) & kmask1) << 4)));

        float s = 0;
        for (int l = 0; l < n; ++l) {
            s += y[l+ 0] * ((int8_t)((q[l+ 0] >> shift) & m3) - ((h[l+ 0] & m) ? 0 : m4));
        }
        float d = d_all * s;
        sumf1 += d * scales[0];
        sumf2 += d;
        //sumf += d_all * s * (scales[0] - 32);

        s = 0;
        for (int l = 0; l < n; ++l) {
            s += y[l+16] * ((int8_t)((q[l+16] >> shift) & m3) - ((h[l+16] & m) ? 0 : m4));
        }
        d = d_all * s;
        sumf1 += d * scales[1];
        sumf2 += d;
        //sumf += d_all * s * (scales[1] - 32);

    }

    //sum[ith] = sumf;
    sum[ith] = sumf1 - 32.f*sumf2;

    //
    // Accumulate the sum from all threads in the threadgroup
    //
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%4 == 0) {
        for (int i = 1; i < 4; ++i) sum[ith] += sum[ith + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%16 == 0) {
        for (int i = 4; i < 16; i += 4) sum[ith] += sum[ith + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith == 0) {
        for (int i = 16; i < nth; i += 16) sum[0] += sum[i];
        dst[r1*ne0 + r0] = sum[0];
    }

}

kernel void kernel_mul_mat_q4_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q4_k * x = (device const block_q4_k *) src0 + r0*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    const int nth = tptg.x*tptg.y;
    const int ith = tptg.y*tpitg.x + tpitg.y;

    const int tid = tpitg.y;   // 0...16
    const int il  = tid/4;     // 0...3
    const int ir  = tid - 4*il;// 0...3
    const int n   = 4;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    sum[ith] = 0.0f;

    uchar2 sc1, sc2, sc3, sc4;

    float sumf = 0;
    for (int i = tpitg.x; i < nb; i += tptg.x) {

        device const uint8_t * q1 = (x + i)->qs + q_offset;
        device const uint8_t * q2 = q1 + 64;
        device const float   * y1 = yy + i*QK_K + y_offset;
        device const float   * y2 = y1 + 128;

        const float dall = (float)((x + i)->d);
        const float dmin = (float)((x + i)->dmin);

        device const uint16_t * a = (device const uint16_t *)(x + i)->scales;
        sc1 = as_type<uchar2>((uint16_t)(a[im+0] & kmask1));
        sc2 = as_type<uchar2>((uint16_t)(a[im+2] & kmask1));
        sc3 = as_type<uchar2>((uint16_t)(((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2)));
        sc4 = as_type<uchar2>((uint16_t)(((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2)));

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < n; ++l) {

            s[0] += y1[l] * (q1[l] & 0xF); s[1] += y1[l+32] * (q1[l] >> 4);
            s[2] += y2[l] * (q2[l] & 0xF); s[3] += y2[l+32] * (q2[l] >> 4);
            smin += y1[l] * sc2[0] + y1[l+32] * sc2[1] + y2[l] * sc4[0] + y2[l+32] * sc4[1];

        }
        sumf += dall * (s[0] * sc1[0] + s[1] * sc1[1] + s[2] * sc3[0] + s[3] * sc3[1]) - dmin * smin;

    }

    sum[ith] = sumf;

    //
    // Accumulate the sum from all threads in the threadgroup
    // This version is slightly faster than the commented out one below,
    // which I copy-pasted from ggerganov's q4_0 dot product for metal.
    //
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%4 == 0) {
        for (int i = 1; i < 4; ++i) sum[ith] += sum[ith + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%16 == 0) {
        for (int i = 4; i < 16; i += 4) sum[ith] += sum[ith + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith == 0) {
        for (int i = 16; i < nth; i += 16) sum[0] += sum[i];
        dst[r1*ne0 + r0] = sum[0];
    }

    //// accumulate the sum from all threads in the threadgroup
    //threadgroup_barrier(mem_flags::mem_threadgroup);
    //for (uint i = nth/2; i > 0; i /= 2) {
    //    if (ith < i) {
    //        sum[ith] += sum[ith + i];
    //    }
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    //if (ith == 0) {
    //    dst[r1*ne0 + r0] = sum[0];
    //}
}

kernel void kernel_mul_mat_q5_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q5_k * x = (device const block_q5_k *) src0 + r0*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    const int nth = tptg.x*tptg.y;
    const int ith = tptg.y*tpitg.x + tpitg.y;

    const int tid = tpitg.y;   // 0...16
    const int il  = tid/4;     // 0...3
    const int ir  = tid - 4*il;// 0...3
    const int n   = 4;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1 = 1u << (2*im);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uchar2 sc1, sc2, sc3, sc4;

    float sumf = 0;
    for (int i = tpitg.x; i < nb; i += tptg.x) {

        device const uint8_t * q1 = (x + i)->qs + q_offset;
        device const uint8_t * q2 = q1 + 64;
        device const uint8_t * qh = (x + i)->qh + l0;
        device const float   * y1 = yy + i*QK_K + y_offset;
        device const float   * y2 = y1 + 128;

        const float dall = (float)((x + i)->d);
        const float dmin = (float)((x + i)->dmin);

        device const uint16_t * a = (device const uint16_t *)(x + i)->scales;
        sc1 = as_type<uchar2>((uint16_t)(a[im+0] & kmask1));
        sc2 = as_type<uchar2>((uint16_t)(a[im+2] & kmask1));
        sc3 = as_type<uchar2>((uint16_t)(((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2)));
        sc4 = as_type<uchar2>((uint16_t)(((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2)));

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < n; ++l) {

            s[0] += y1[l+ 0] * ((q1[l] & 0xF) + (qh[l] & hm1 ? 16 : 0));
            s[1] += y1[l+32] * ((q1[l] >>  4) + (qh[l] & hm2 ? 16 : 0));
            s[2] += y2[l+ 0] * ((q2[l] & 0xF) + (qh[l] & hm3 ? 16 : 0));
            s[3] += y2[l+32] * ((q2[l] >>  4) + (qh[l] & hm4 ? 16 : 0));
            smin += y1[l] * sc2[0] + y1[l+32] * sc2[1] + y2[l] * sc4[0] + y2[l+32] * sc4[1];

        }
        sumf += dall * (s[0] * sc1[0] + s[1] * sc1[1] + s[2] * sc3[0] + s[3] * sc3[1]) - dmin * smin;

    }
    sum[ith] = sumf;

    //
    // Accumulate the sum from all threads in the threadgroup
    //
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%4 == 0) {
        sum[ith] += sum[ith+1] + sum[ith+2] + sum[ith+3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%16 == 0) {
        sum[ith] += sum[ith+4] + sum[ith+8] + sum[ith+12];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith == 0) {
        for (int i = 16; i < nth; i += 16) sum[0] += sum[i];
        dst[r1*ne0 + r0] = sum[0];
    }

}

kernel void kernel_mul_mat_q6_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne10,
        constant   int64_t & ne0,
        threadgroup float  * sum [[threadgroup(0)]],
        uint2 tgpig[[threadgroup_position_in_grid]],
        uint2 tpitg[[thread_position_in_threadgroup]],
        uint2  tptg[[threads_per_threadgroup]]) {

    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;

    device const block_q6_k * x = (device const block_q6_k *) src0 + r0*nb;
    device const float     * yy = (device const float      *) src1 + r1*ne10;

    const int nth = tptg.x*tptg.y;
    const int ith = tptg.y*tpitg.x + tpitg.y;

    // Note: we absolutely assume that tptg.y = 16 and QK_K = 256!
    const int iqs  = 16 * tpitg.y;
    const int ip   = iqs / 128;         // 0 or 1
    const int il   = (iqs - 128*ip)/16; // 0...7
    const int n    = 4;
    const int l0   = n*il;
    const int is   = 8*ip + l0/16;

    const int y_offset = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    float sumf = 0;
    for (int i = tpitg.x; i < nb; i += tptg.x) {

        device const uint8_t * ql = x[i].ql + q_offset_l;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((ql[l+ 0] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+32] * ((int8_t)((ql[l+32] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+64] * ((int8_t)((ql[l+ 0]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += y[l+96] * ((int8_t)((ql[l+32]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

    }

    sum[ith] = sumf;

    //
    // Accumulate the sum from all threads in the threadgroup
    //
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%4 == 0) {
        for (int i = 1; i < 4; ++i) sum[ith] += sum[ith + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith%16 == 0) {
        for (int i = 4; i < 16; i += 4) sum[ith] += sum[ith + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ith == 0) {
        for (int i = 16; i < nth; i += 16) sum[0] += sum[i];
        dst[r1*ne0 + r0] = sum[0];
    }

}
