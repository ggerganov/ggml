#pragma once

#include "common.cuh"
#include "vecdotq.cuh"
#include "mma.cuh"

#include <climits>
#include <cstdint>

#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE/QI8_1)

typedef void (*load_tiles_mmq_t)(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride);
typedef void (*vec_dot_mmq_t)(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0);
typedef void (*mmq_write_back_t)(const float * __restrict__ sum, float * __restrict__ dst, const int & ne0, const int & ne1);

struct block_q8_1_mmq {
    half2  ds[4];
    int8_t qs[4*QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

// get_mmq_x_max_host is in common.cuh so that it can be used to determine the correct way to round for --split-mode row

static constexpr __device__ int get_mmq_x_max_device() {
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    return 64;
#else
#if __CUDA_ARCH__ >= CC_VOLTA
#ifdef CUDA_USE_TENSOR_CORES
    return MMQ_MAX_BATCH_SIZE;
#else
    return 128;
#endif // CUDA_USE_TENSOR_CORES
#else
    return 64;
#endif // __CUDA_ARCH__ >= CC_VOLTA
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
}

// get_mmq_y_host is in common.cuh so that it can be used to determine the correct way to round for --split-mode row

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
static constexpr __device__ int get_mmq_y_device(int mmq_x) {
    return mmq_x >= 32 ? 128 : 64;
}
#else
#if __CUDA_ARCH__ >= CC_VOLTA
static constexpr __device__ int get_mmq_y_device(int mmq_x) {
    return mmq_x >= 32 ? 128 : 64;
}
#else
static constexpr __device__ int get_mmq_y_device(int /*mmq_x*/) {
    return 64;
}
#endif // __CUDA_ARCH__ >= CC_VOLTA
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)

#define TILE_X_SIZES_Q4_0 tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_0 + mmq_y/QI4_0, 0}
#define TILE_X_SIZES_Q4_1 tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_1 + mmq_y/QI4_1, 0}
#define TILE_X_SIZES_Q5_0 tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_0 + mmq_y/QI5_0, 0}
#define TILE_X_SIZES_Q5_1 tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_1 + mmq_y/QI5_1, 0}
#define TILE_X_SIZES_Q8_0 tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI8_0 + mmq_y/QI8_0, 0}
#define TILE_X_SIZES_Q2_K tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE       + mmq_y,       0}
#define TILE_X_SIZES_Q3_K tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI3_K + mmq_y/QI3_K, mmq_y*WARP_SIZE/4 + mmq_y/4}
#define TILE_X_SIZES_Q4_K tile_x_sizes{mmq_y*WARP_SIZE   + mmq_y, mmq_y*WARP_SIZE/QI4_K + mmq_y/QI4_K, mmq_y*WARP_SIZE/8 + mmq_y/8}
#define TILE_X_SIZES_Q5_K tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI5_K + mmq_y/QI5_K, mmq_y*WARP_SIZE/8 + mmq_y/8}
#define TILE_X_SIZES_Q6_K tile_x_sizes{mmq_y*WARP_SIZE*2 + mmq_y, mmq_y*WARP_SIZE/QI6_K + mmq_y/QI6_K, mmq_y*WARP_SIZE/8 + mmq_y/8}

#define GET_TILE_X_SIZES_BODY                           \
    return type == GGML_TYPE_Q4_0 ? TILE_X_SIZES_Q4_0 : \
        type == GGML_TYPE_Q4_1 ? TILE_X_SIZES_Q4_1 :    \
        type == GGML_TYPE_Q5_0 ? TILE_X_SIZES_Q5_0 :    \
        type == GGML_TYPE_Q5_1 ? TILE_X_SIZES_Q5_1 :    \
        type == GGML_TYPE_Q8_0 ? TILE_X_SIZES_Q8_0 :    \
        type == GGML_TYPE_Q2_K ? TILE_X_SIZES_Q2_K :    \
        type == GGML_TYPE_Q3_K ? TILE_X_SIZES_Q3_K :    \
        type == GGML_TYPE_Q4_K ? TILE_X_SIZES_Q4_K :    \
        type == GGML_TYPE_Q5_K ? TILE_X_SIZES_Q5_K :    \
        type == GGML_TYPE_Q6_K ? TILE_X_SIZES_Q6_K :    \
        tile_x_sizes{0, 0, 0}

static tile_x_sizes get_tile_x_sizes_host(const ggml_type type, const int mmq_y) {
    GET_TILE_X_SIZES_BODY;
}

template <int mmq_y>
static constexpr __device__ tile_x_sizes get_tile_x_sizes_device(ggml_type type) {
    GET_TILE_X_SIZES_BODY;
}

// ------------------------------------------------------------

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

    const int kbx  = threadIdx.x / QI4_0;
    const int kqsx = threadIdx.x % QI4_0;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;

        x_qs[i * (WARP_SIZE + 1) + threadIdx.x] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + threadIdx.y * QI4_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));

            int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
            for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI4_0) % WARP_SIZE];
            }

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE + 1) + k0], u, x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + k0/QI4_0],
                y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_0_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    GGML_UNUSED(x_sc);

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    mma_A A;
    float dA[mma_C::ne/2];

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

#pragma unroll
    for (int l = 0; l < mma_A::ne; ++l) {
        const int i     = i0 + mma_A::get_i(l);
        const int k     = k0 + mma_A::get_k(l) % QI4_0;
        const int shift =   4*(mma_A::get_k(l) / QI4_0);

        A.x[l] = __vsubss4((x_qs[i*(WARP_SIZE + 1) + k] >> shift) & 0x0F0F0F0F, 0x08080808);
    }
#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dA[l] = x_df[i*(WARP_SIZE/QI4_0) + i/QI4_0 + k0/QI4_0];
    }

    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        mma_C C;
        mma_B B;
        half2 dsB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_B::ne; ++l) {
            const int j =    j0 + mma_B::get_j(l);
            const int k = (2*k0 + mma_B::get_k(l)) % WARP_SIZE;

            B.x[l] = y_qs[j*MMQ_TILE_Y_K + k];
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dsB[l] = y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)];
        }

        C.mma_K8(A, B);

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            sum[(j0/B.J)*C.ne + l] += dA[l/2]*__low2float(dsB[l%2])*C.x[l];
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

    const int kbx  = threadIdx.x / QI4_1;
    const int kqsx = threadIdx.x % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbx;

        x_qs[i * (WARP_SIZE + 1) + threadIdx.x] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + threadIdx.y * QI4_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *) x + kbx0 + i*stride + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));

            int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
            for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI4_1) % WARP_SIZE];
            }

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE + 1) + k0], u, x_dm[i*(WARP_SIZE/QI4_1) + i/QI4_1 + k0/QI4_1],
                y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_1_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    GGML_UNUSED(x_sc);

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    mma_A A;
    half2 dmA[mma_C::ne/2];

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

#pragma unroll
    for (int l = 0; l < mma_A::ne; ++l) {
        const int i     = i0 + mma_A::get_i(l);
        const int k     = k0 + mma_A::get_k(l) % QI4_0;
        const int shift =   4*(mma_A::get_k(l) / QI4_0);

        A.x[l] = (x_qs[i*(WARP_SIZE + 1) + k] >> shift) & 0x0F0F0F0F;
    }
#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dmA[l] = x_dm[i*(WARP_SIZE/QI4_0) + i/QI4_0 + k0/QI4_0];
    }

    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        mma_C C;
        mma_B B;
        half2 dsB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_B::ne; ++l) {
            const int j =    j0 + mma_B::get_j(l);
            const int k = (2*k0 + mma_B::get_k(l)) % WARP_SIZE;

            B.x[l] = y_qs[j*MMQ_TILE_Y_K + k];
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dsB[l] = y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)];
        }

        C.mma_K8(A, B);

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            const half2 dmA_dsB = dmA[l/2]*dsB[l%2];
            sum[(j0/B.J)*C.ne + l] += __low2float(dmA_dsB)*C.x[l] + __high2float(dmA_dsB);
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

    const int kbx  = threadIdx.x / QI5_0;
    const int kqsx = threadIdx.x % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        x_qs[i * (2*WARP_SIZE + 1) + 2*threadIdx.x+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_qs[i * (2*WARP_SIZE + 1) + 2*threadIdx.x+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + threadIdx.y * QI5_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_0_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const float * x_dmf = (const float *) x_dm;
    const int   * y_qs  = (const int   *) y + 4;
    const float * y_df  = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));
            const int index_bx = i*(WARP_SIZE/QI5_0) + i/QI5_0 + k0/QI5_0;

            int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
            for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI5_0) % WARP_SIZE];
            }

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_0_q8_1_impl<float, QR5_0*VDR_Q5_0_Q8_1_MMQ>
                (&x_qs[i*(2*WARP_SIZE + 1) + 2*k0], u, x_dmf[index_bx], y_df[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_0_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    GGML_UNUSED(x_sc);

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    mma_A A;
    float dA[mma_C::ne/2];

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

#pragma unroll
    for (int l = 0; l < mma_A::ne; ++l) {
        const int i     =    i0 + mma_A::get_i(l);
        const int k     = 2*(k0 + mma_A::get_k(l) % QI5_0) + mma_A::get_k(l) / QI5_0;

        A.x[l] = x_qs[i*(2*WARP_SIZE + 1) + k];
    }
#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dA[l] = x_df[i*(WARP_SIZE/QI5_0) + i/QI5_0 + k0/QI5_0];
    }

    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        mma_C C;
        mma_B B;
        float dB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_B::ne; ++l) {
            const int j =    j0 + mma_B::get_j(l);
            const int k = (2*k0 + mma_B::get_k(l)) % WARP_SIZE;

            B.x[l] = y_qs[j*MMQ_TILE_Y_K + k];
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dB[l] = y_df[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)];
        }

        C.mma_K8(A, B);

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            sum[(j0/B.J)*C.ne + l] += dA[l/2]*dB[l%2]*C.x[l];
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

    const int kbx  = threadIdx.x / QI5_1;
    const int kqsx = threadIdx.x % QI5_1;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (threadIdx.x % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_qs[i * (2*WARP_SIZE + 1) + 2*threadIdx.x+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_qs[i * (2*WARP_SIZE + 1) + 2*threadIdx.x+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + threadIdx.y * QI5_1 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *) x + kbx0 + i*stride + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_1_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const int   * y_qs  = (const int   *) y + 4;
    const half2 * y_ds  = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const int kyqs = k0 % (QI8_1/2) + QI8_1 * (k0 / (QI8_1/2));
            const int index_bx = i*(WARP_SIZE/QI5_1) + i/QI5_1 + k0/QI5_1;

            int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
            for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
                u[2*l+0] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l)         % WARP_SIZE];
                u[2*l+1] = y_qs[j*MMQ_TILE_Y_K + (kyqs + l + QI5_1) % WARP_SIZE];
            }

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                (&x_qs[i*(2*WARP_SIZE + 1) + 2*k0], u, x_dm[index_bx], y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_1_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    GGML_UNUSED(x_sc);

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    mma_A A;
    half2 dmA[mma_C::ne/2];

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

#pragma unroll
    for (int l = 0; l < mma_A::ne; ++l) {
        const int i     =    i0 + mma_A::get_i(l);
        const int k     = 2*(k0 + mma_A::get_k(l) % QI5_1) + mma_A::get_k(l) / QI5_1;

        A.x[l] = x_qs[i*(2*WARP_SIZE + 1) + k];
    }
#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dmA[l] = x_dm[i*(WARP_SIZE/QI5_1) + i/QI5_1 + k0/QI5_1];
    }

    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        mma_C C;
        mma_B B;
        half2 dsB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_B::ne; ++l) {
            const int j =    j0 + mma_B::get_j(l);
            const int k = (2*k0 + mma_B::get_k(l)) % WARP_SIZE;

            B.x[l] = y_qs[j*MMQ_TILE_Y_K + k];
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dsB[l] = y_ds[j*MMQ_TILE_Y_K + (2*k0/QI8_1) % (WARP_SIZE/QI8_1)];
        }

        C.mma_K8(A, B);

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            const half2 dmA_dsB = dmA[l/2]*dsB[l%2];
            sum[(j0/B.J)*C.ne + l] += __low2float(dmA_dsB)*C.x[l] + __high2float(dmA_dsB);
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {
    GGML_UNUSED(x_sc);

    const int kbx  = threadIdx.x / QI8_0;
    const int kqsx = threadIdx.x % QI8_0;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;

        x_qs[i * (WARP_SIZE + 1) + threadIdx.x] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + threadIdx.y * QI8_0 + threadIdx.x / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
    GGML_UNUSED(x_sc);

    const float * x_dmf = (const float *) x_dm;
    const int   * y_qs  = (const int   *) y + 4;
    const float * y_df  = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                (&x_qs[i*(WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0], x_dmf[i*(WARP_SIZE/QI8_0) + i/QI8_0 + k0/QI8_0],
                y_df[j*MMQ_TILE_Y_K + k0/QI8_1]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE
    GGML_UNUSED(x_sc);

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    mma_A A;
    float dA[mma_C::ne/2];

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

#pragma unroll
    for (int l = 0; l < mma_A::ne; ++l) {
        const int i = i0 + mma_A::get_i(l);
        const int k = k0 + mma_A::get_k(l);

        A.x[l] = x_qs[i*(WARP_SIZE + 1) + k];
    }
#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dA[l] = x_df[i*(WARP_SIZE/QI8_0) + i/QI8_0 + k0/QI8_0];
    }

    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        mma_C C;
        mma_B B;
        float dB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_B::ne; ++l) {
            const int j = j0 + mma_B::get_j(l);
            const int k = k0 + mma_B::get_k(l);

            B.x[l] = y_qs[j*MMQ_TILE_Y_K + k];
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dB[l] = y_df[j*MMQ_TILE_Y_K + k0/QI8_1];
        }

        C.mma_K8(A, B);

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            sum[(j0/B.J)*C.ne + l] += C.x[l]*dA[l/2]*dB[l%2];
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

    const int kbx  = threadIdx.x / QI2_K;
    const int kqsx = threadIdx.x % QI2_K;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *) x + kbx0 + i*stride + kbx;

        const int x_ql_0 = get_int_from_uint8(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = kbx*QI2_K + (kqsx/8)*8 + l*2 + (kqsx % 8)/4;

            int x_qs_k = ((x_ql_0 >> (2*l)) & 0x03030303) << (2*(kqsx % 4));
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 1, WARP_SIZE);
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 2, WARP_SIZE);

            if (kqsx % QR2_K != 0) {
                continue;
            }

            x_qs[i*(WARP_SIZE + 1) + k] = x_qs_k;
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

        x_dm[i*(WARP_SIZE + 1) + threadIdx.x] = x_dm_ik;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q2_K_q8_1_impl_mmq(
                &x_qs[i*(WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + (QR2_K*k0) % WARP_SIZE],
                &x_dm[i*(WARP_SIZE + 1) + k0], y_df[j*MMQ_TILE_Y_K + ((QR2_K*k0) % WARP_SIZE)/QI8_1]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q2_K_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

    mma_A   A[2];
    float  dA[mma_C::ne/2][2];
    float  mA[mma_C::ne/2][2];

#pragma unroll
    for (int l = 0; l < mma_A::ne; ++l) {
        const int i = i0 + mma_A::get_i(l);
        const int shift = 2*mma_A::get_k(l);

        A[0].x[l] = (x_qs[i*(WARP_SIZE + 1) + k0 + 0] >> shift) & 0x03030303;
        A[1].x[l] = (x_qs[i*(WARP_SIZE + 1) + k0 + 1] >> shift) & 0x03030303;
    }

#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const float2 dm = __half22float2(x_dm[i*(WARP_SIZE + 1) + k0 + kk]);

            dA[l][kk] = dm.x;
            mA[l][kk] = dm.y;
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        mma_C Cd[2];
        mma_C Cm[2];
        mma_B B[2];
        float dB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_B::ne; ++l) {
            const int j = j0 + mma_B::get_j(l);
            const int k = (4*k0 + mma_B::get_k(l)) % WARP_SIZE;

            B[0].x[l] = y_qs[j*MMQ_TILE_Y_K + k + 0];
            B[1].x[l] = y_qs[j*MMQ_TILE_Y_K + k + mma_B::K];
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dB[l] = y_df[j*MMQ_TILE_Y_K + ((4*k0)/QI8_1) % (WARP_SIZE/QI8_1)];
        }

        Cd[0].mma_K4(A[0], B[0]);
        Cd[1].mma_K4(A[1], B[1]);

        mma_A A1;
        A1.x[0] = 0x01010101;
        A1.x[1] = 0x01010101;
        Cm[0].mma_K4(A1, B[0]);
        Cm[1].mma_K4(A1, B[1]);

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            sum[(j0/mma_B::J)*mma_C::ne + l] += (Cd[0].x[l]*dA[l/2][0] + Cd[1].x[l]*dA[l/2][1] - Cm[0].x[l]*mA[l/2][0] - Cm[1].x[l]*mA[l/2][1])*dB[l%2];
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

    const int kbx  = threadIdx.x / QI3_K;
    const int kqsx = threadIdx.x % QI3_K;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride + kbx;

        const int x_ql_0 = get_int_from_uint8(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_from_uint8(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = kbx*(QR3_K*QI3_K) + (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            int x_qs_k = (x_ql_k | x_qh_k) << (4*(k%2));
            x_qs_k |= __shfl_xor_sync(0xFFFFFFFF, x_qs_k, 1, WARP_SIZE);

            if (kqsx % 2 != 0) {
                continue;
            }

            x_qs[i*(2*WARP_SIZE + 1) + k/2] = x_qs_k;
        }
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + threadIdx.y * QI3_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = threadIdx.x % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE/4) + i / 4 + threadIdx.x % (WARP_SIZE/4)] = sc;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const int kbx  = k0 / QI3_K;
            const int ky  = (k0 % QI3_K) * QR3_K;

            const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q3_K_q8_1_impl_mmq(
                &x_qs[i*(2*WARP_SIZE + 1) + 2*k0], &y_qs[j*MMQ_TILE_Y_K + (k0*QR3_K) % WARP_SIZE], scales,
                x_df[i*(WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[j*MMQ_TILE_Y_K + ((k0*QR3_K) % WARP_SIZE)/QI8_1]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q3_K_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

    mma_A   A[2];
    int   scA[mma_C::ne/2][2];
    float  dA[mma_C::ne/2];

#pragma unroll
    for (int l = 0; l < mma_A::ne; ++l) {
        const int i = i0 + mma_A::get_i(l);
        const int k = QR3_K*k0 + mma_A::get_k(l);

        A[0].x[l] = (x_qs[i*(2*WARP_SIZE + 1) + k/2 + 0]          >> (4*(k%2))) & 0x0F0F0F0F;
        A[1].x[l] = (x_qs[i*(2*WARP_SIZE + 1) + k/2 + mma_A::K/2] >> (4*(k%2))) & 0x0F0F0F0F;
        A[0].x[l] = __vsubss4(A[0].x[l], 0x04040404);
        A[1].x[l] = __vsubss4(A[1].x[l], 0x04040404);
    }

#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        const int kbx  = k0 / QI3_K;
        const int ky  = (k0 % QI3_K) * QR3_K;
        const int8_t * sc = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

        scA[l][0] = sc[0];
        scA[l][1] = sc[1];
    }

#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dA[l] = x_df[i*(WARP_SIZE/QI3_K) + i/QI3_K + k0/QI3_K];
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        mma_C C[2];
        mma_B B[2];
        float dB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_B::ne; ++l) {
            const int j = j0 + mma_B::get_j(l);
            const int k = (4*k0 + mma_B::get_k(l)) % WARP_SIZE;

            B[0].x[l] = y_qs[j*MMQ_TILE_Y_K + k + 0];
            B[1].x[l] = y_qs[j*MMQ_TILE_Y_K + k + mma_B::K];
        }
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dB[l] = y_df[j*MMQ_TILE_Y_K + ((4*k0)/QI8_1) % (WARP_SIZE/QI8_1)];
        }

        C[0].mma_K4(A[0], B[0]);
        C[1].mma_K4(A[1], B[1]);

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            sum[(j0/mma_B::J)*mma_C::ne + l] += (C[0].x[l]*scA[l/2][0] + C[1].x[l]*scA[l/2][1])*dA[l/2]*dB[l%2];
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

    const int kbx  = 0;           // threadIdx.x / QI4_K
    const int kqsx = threadIdx.x; // threadIdx.x % QI4_K

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + kbx;

        x_qs[i * (WARP_SIZE + 1) + threadIdx.x] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + threadIdx.y * QI4_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + kbxd;

        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/16]) + 2*((k0 % 16) / 8);

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_K_q8_1_impl_mmq(
                &x_qs[i*(WARP_SIZE + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + (QR4_K*k0) % WARP_SIZE], sc, sc+8,
                x_dm[i*(WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[j*MMQ_TILE_Y_K + ((QR4_K*k0) % WARP_SIZE)/QI8_1]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

    mma_A   A[2];
    int   scA[mma_C::ne/2][2];
    int    mA[mma_C::ne/2][2];
    half2 dmA[mma_C::ne/2];
#pragma unroll
    for (int kvdr = 0; kvdr < VDR_Q4_K_Q8_1_MMQ; kvdr += 4) {
#pragma unroll
        for (int l = 0; l < mma_A::ne; ++l) {
            const int i = i0 + mma_A::get_i(l);
            const int k = k0 + mma_A::get_k(l);

            A[kvdr/4].x[l] = (x_qs[i*(WARP_SIZE + 1) + k] >> kvdr) & 0x0F0F0F0F;
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + mma_C::get_i(2*l);

            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/16]) + 2 * ((k0 % 16) / 8);
            const uint8_t *  m = sc + 8;

            scA[l][kvdr/4] = sc[kvdr/4];
            mA[l][kvdr/4]  =  m[kvdr/4];
        }
    }

#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dmA[l] = x_dm[i*(WARP_SIZE/QI5_K) + i/QI5_K + k0/QI5_K];
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        float tmpd[mma_C::ne] = {0.0f};
        float tmpm[mma_C::ne] = {0.0f};

#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q5_K_Q8_1_MMQ; kvdr += 4) {
            mma_C   C;
            mma_B   B;
            half2 dsB[mma_C::ne/2];

#pragma unroll
            for (int l = 0; l < mma_B::ne; ++l) {
                const int j = j0 + mma_B::get_j(l);
                const int k = (2*k0 + 2*kvdr + mma_B::get_k(l)) % WARP_SIZE;

                B.x[l] = y_qs[j*MMQ_TILE_Y_K + k];
            }
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dsB[l] = y_ds[j*MMQ_TILE_Y_K + ((2*k0 + 2*kvdr)/QI8_1) % (WARP_SIZE/QI8_1)];
            }

            C.mma_K8(A[kvdr/4], B);

#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                tmpd[l] += (C.x[l]*scA[l/2][kvdr/4]) *  __low2float(dsB[l%2]);
                tmpm[l] += mA[l/2][kvdr/4]           * __high2float(dsB[l%2]);
            }
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            sum[(j0/mma_B::J)*mma_C::ne + l] += __low2float(dmA[l/2])*tmpd[l] - __high2float(dmA[l/2])*tmpm[l];
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

    const int kbx  = 0;           // threadIdx.x / QI5_K
    const int kqsx = threadIdx.x; // threadIdx.x % QI5_K

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + threadIdx.x % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + threadIdx.x % (QI5_K/4) + (QI5_K/4);

        x_qs[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_qs[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + threadIdx.y * QI5_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride + kbxd;

        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const int   * y_qs  = (const int   *) y + 4;
    const half2 * y_ds  = (const half2 *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/16]) + 2 * ((k0 % 16) / 8);

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q5_K_q8_1_impl_mmq(
                &x_qs[i*(QR5_K*WARP_SIZE + 1) + QR5_K*k0], &y_qs[j*MMQ_TILE_Y_K + (QR5_K*k0) % WARP_SIZE], sc, sc+8,
                x_dm[i*(WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[j*MMQ_TILE_Y_K + ((QR5_K*k0) % WARP_SIZE)/QI8_1]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K8 mma_A;
    typedef mma_int_B_J8K8  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = threadIdx.y*mma_A::I;
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");

    mma_A   A[2];
    int   scA[mma_C::ne/2][2];
    int    mA[mma_C::ne/2][2];
    half2 dmA[mma_C::ne/2];
#pragma unroll
    for (int kvdr = 0; kvdr < VDR_Q5_K_Q8_1_MMQ; kvdr += 4) {
#pragma unroll
        for (int l = 0; l < mma_A::ne; ++l) {
            const int i = i0 + mma_A::get_i(l);
            const int k = QR5_K*k0 + QR5_K*kvdr + mma_A::get_k(l);

            A[kvdr/4].x[l] = x_qs[i*(QR5_K*WARP_SIZE + 1) + k];
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + mma_C::get_i(2*l);

            const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/16]) + 2 * ((k0 % 16) / 8);
            const uint8_t *  m = sc + 8;

            scA[l][kvdr/4] = sc[kvdr/4];
            mA[l][kvdr/4]  =  m[kvdr/4];
        }
    }

#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dmA[l] = x_dm[i*(WARP_SIZE/QI5_K) + i/QI5_K + k0/QI5_K];
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        float tmpd[mma_C::ne] = {0.0f};
        float tmpm[mma_C::ne] = {0.0f};

#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q5_K_Q8_1_MMQ; kvdr += 4) {
            mma_C   C;
            mma_B   B;
            half2 dsB[mma_C::ne/2];

#pragma unroll
            for (int l = 0; l < mma_B::ne; ++l) {
                const int j = j0 + mma_B::get_j(l);
                const int k = (2*k0 + 2*kvdr + mma_B::get_k(l)) % WARP_SIZE;

                B.x[l] = y_qs[j*MMQ_TILE_Y_K + k];
            }
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dsB[l] = y_ds[j*MMQ_TILE_Y_K + ((2*k0 + 2*kvdr)/QI8_1) % (WARP_SIZE/QI8_1)];
            }

            C.mma_K8(A[kvdr/4], B);

#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                tmpd[l] += (C.x[l]*scA[l/2][kvdr/4]) *  __low2float(dsB[l%2]);
                tmpm[l] += mA[l/2][kvdr/4]           * __high2float(dsB[l%2]);
            }
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            sum[(j0/mma_B::J)*mma_C::ne + l] += __low2float(dmA[l/2])*tmpd[l] - __high2float(dmA[l/2])*tmpm[l];
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const char * __restrict__ x, int * __restrict__ x_qs, half2 * __restrict__ x_dm,
    int * __restrict__ x_sc, const int & kbx0, const int & i_max, const int & stride) {

    const int kbx  = 0;           // threadIdx.x / QI6_K
    const int kqsx = threadIdx.x; // threadIdx.x % QI6_K

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + threadIdx.x % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + threadIdx.x % (QI6_K/2) + (QI6_K/2);

        x_qs[i * (2*WARP_SIZE + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i * (2*WARP_SIZE + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K;  // == 1 if QK_K == 256
    const int kbxd = threadIdx.x % blocks_per_tile_x_row; // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + threadIdx.y * QI6_K + threadIdx.x / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + threadIdx.x % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, threadIdx.x % (QI6_K/8));
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_dp4a(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {

    const float * x_dmf = (const float *) x_dm;
    const int   * y_qs  = (const int   *) y + 4;
    const float * y_df  = (const float *) y;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/8]);

            sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q6_K_q8_1_impl_mmq(
                &x_qs[i*(QR6_K*WARP_SIZE + 1) + QR6_K*k0], &y_qs[j*MMQ_TILE_Y_K + (QR6_K*k0) % WARP_SIZE], sc,
                x_dmf[i*(WARP_SIZE/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + ((QR6_K*k0) % WARP_SIZE)/QI8_1]);
        }
    }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(
    const int * __restrict__ x_qs, const half2 * __restrict__ x_dm, const int * __restrict__ x_sc,
    const int * __restrict__ y, float * __restrict__ sum, const int & k0) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    const float * x_df = (const float *) x_dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = threadIdx.y*mma_A::I;
#ifdef INT8_MMA_AVAILABLE
    static_assert(nwarps*mma_A::I == mmq_y, "nwarps*mma_A::I != mmq_y");
#endif // INT8_MMA_AVAILABLE

    mma_A   A[4];
    int   scA[mma_C::ne/2][4];
    float  dA[mma_C::ne/2];
#pragma unroll
    for (int kvdr = 0; kvdr < VDR_Q6_K_Q8_1_MMQ; kvdr += 4) {
#pragma unroll
        for (int l = 0; l < mma_A::ne; ++l) {
            const int i = i0 + mma_A::get_i(l);
            const int k = QR6_K*k0 + QR6_K*kvdr + mma_A::get_k(l);

            A[kvdr/2 + 0].x[l] = x_qs[i*(QR6_K*WARP_SIZE + 1) + k + 0];
            A[kvdr/2 + 1].x[l] = x_qs[i*(QR6_K*WARP_SIZE + 1) + k + mma_A::K];
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + mma_C::get_i(2*l);

            const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/8]);

            scA[l][kvdr/2 + 0] = sc[kvdr/2 + 0];
            scA[l][kvdr/2 + 1] = sc[kvdr/2 + 1];
        }
    }

#pragma unroll
    for (int l = 0; l < mma_C::ne/2; ++l) {
        const int i = i0 + mma_C::get_i(2*l);

        dA[l] = x_df[i*(WARP_SIZE/QI6_K) + i/QI6_K + k0/QI6_K];
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += mma_int_B_J8K8::J) {
        float tmp[mma_C::ne] = {0.0f};

#pragma unroll
        for (int kvdr = 0; kvdr < VDR_Q6_K_Q8_1_MMQ; kvdr += 4) {
            mma_C C[2];
            mma_B B[2];
            float dB[mma_C::ne/2];

#pragma unroll
            for (int l = 0; l < mma_B::ne; ++l) {
                const int j = j0 + mma_B::get_j(l);
                const int k = (2*k0 + 2*kvdr + mma_B::get_k(l)) % WARP_SIZE;

                B[0].x[l] = y_qs[j*MMQ_TILE_Y_K + k + 0];
                B[1].x[l] = y_qs[j*MMQ_TILE_Y_K + k + mma_B::K];
            }
#pragma unroll
            for (int l = 0; l < mma_C::ne/2; ++l) {
                const int j = j0 + mma_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + ((2*k0 + 2*kvdr)/QI8_1) % (WARP_SIZE/QI8_1)];
            }

            C[0].mma_K4(A[kvdr/2 + 0], B[0]);
            C[1].mma_K4(A[kvdr/2 + 1], B[1]);

#pragma unroll
            for (int l = 0; l < mma_C::ne; ++l) {
                tmp[l] += (C[0].x[l]*scA[l/2][kvdr/2 + 0] + C[1].x[l]*scA[l/2][kvdr/2 + 1])*dB[l%2];
            }
        }

#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            sum[(j0/mma_B::J)*mma_C::ne + l] += tmp[l]*dA[l/2];
        }
    }
#else
    GGML_UNUSED(x_qs); GGML_UNUSED(x_dm); GGML_UNUSED(x_sc); GGML_UNUSED(y); GGML_UNUSED(sum); GGML_UNUSED(k0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(const float * __restrict__ sum, float * __restrict__ dst, const int & ne0, const int & ne1) {
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = blockIdx.y*mmq_x + j0 + threadIdx.y;

        if (j >= ne1) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = blockIdx.x*mmq_y + i0 + threadIdx.x;

            if (need_check && i >= ne0) {
                continue;
            }

            dst[j*ne0 + i] = sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(const float * __restrict__ sum, float * __restrict__ dst, const int & ne0, const int & ne1) {
    typedef mma_int_C_I16J8 mma_C;

    const int i0 = threadIdx.y*mma_C::I;
#ifdef INT8_MMA_AVAILABLE
    static_assert(nwarps*mma_C::I == mmq_y, "nwarps*mma_C::I != mmq_y");
#endif // INT8_MMA_AVAILABLE

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += mma_C::J) {
#pragma unroll
        for (int l = 0; l < mma_C::ne; ++l) {
            const int j = blockIdx.y*mmq_x + j0 + mma_C::get_j(l);

            if (j >= ne1) {
                continue;
            }

            const int i = blockIdx.x*mmq_y + i0 + mma_C::get_i(l);

            if (need_check && i >= ne0) {
                continue;
            }

            dst[j*ne0 + i] = sum[(j0/mma_C::J)*mma_C::ne + l];
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------

template <int mmq_x, int mmq_y, int nwarps, bool need_check, ggml_type type>
struct mmq_type_traits;

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_0> {
    static constexpr int              vdr          = VDR_Q4_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q4_0_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_1> {
    static constexpr int              vdr          = VDR_Q4_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q4_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_0> {
    static constexpr int              vdr          = VDR_Q5_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q5_0_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_1> {
    static constexpr int              vdr          = VDR_Q5_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_1<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q5_1_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_1_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q8_0> {
    static constexpr int              vdr          = VDR_Q8_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q8_0<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q2_K> {
    static constexpr int              vdr          = VDR_Q2_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q2_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q2_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q2_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q3_K> {
    static constexpr int              vdr          = VDR_Q3_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q3_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q3_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q3_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q4_K> {
    static constexpr int              vdr          = VDR_Q4_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q4_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q5_K> {
    static constexpr int              vdr          = VDR_Q5_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q5_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q5_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q5_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_Q6_K> {
    static constexpr int              vdr          = VDR_Q6_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q6_K<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y, nwarps>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q6_K_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

static bool mmq_need_sum(const ggml_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return true;
        case GGML_TYPE_Q5_0:
            return false;
        case GGML_TYPE_Q5_1:
            return true;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return false;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return true;
        case GGML_TYPE_Q6_K:
            return false;
        default:
            GGML_ASSERT(false);
            break;
    }
    return false;
}

template <ggml_type type, int mmq_x, int nwarps, bool need_check>
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
#if defined(RDNA3) || defined(RDNA2)
    __launch_bounds__(WARP_SIZE*nwarps, 2)
#endif // defined(RDNA3) || defined(RDNA2)
#else
#if __CUDA_ARCH__ >= CC_VOLTA
    __launch_bounds__(WARP_SIZE*nwarps, 1)
#else
    __launch_bounds__(WARP_SIZE*nwarps, 2)
#endif // __CUDA_ARCH__ >= CC_VOLTA
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
static __global__ void mul_mat_q(
    const char * __restrict__ x, const char * __restrict__ yc, float * __restrict__ dst,
    const int ne00, const int ne01, const int stride01, const int ne10, const int ne11, const int stride11, const int ne0) {

    // Skip unused template specializations for faster compilation:
    if (mmq_x > get_mmq_x_max_device()) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              qr         = ggml_cuda_type_traits<type>::qr;
    constexpr int              qi         = ggml_cuda_type_traits<type>::qi;
    constexpr int              mmq_y      = get_mmq_y_device(mmq_x);
    constexpr int              vdr        = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vdr;
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::load_tiles;

#ifdef INT8_MMA_AVAILABLE
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<mmq_x, mmq_y, nwarps, need_check>;
#else
    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, type>::vec_dot_dp4a;
    constexpr mmq_write_back_t write_back = mmq_write_back_dp4a<mmq_x, mmq_y, nwarps, need_check>;
#endif // INT8_MMA_AVAILABLE

    constexpr tile_x_sizes txs = get_tile_x_sizes_device<mmq_y>(type);

    extern __shared__ char data_mul_mat_q[];
    int   * tile_x_qs = (int   *)  data_mul_mat_q;
    half2 * tile_x_dm = (half2 *) (tile_x_qs + txs.qs);
    int   * tile_x_sc = (int   *) (tile_x_dm + txs.dm);
    int   * tile_y    = (int   *) (tile_x_sc + txs.sc); // [mmq_x * (WARP_SIZE + WARP_SIZE/QI8_1)]

    const int blocks_per_row_x = ne00 / qk;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int & ne1 = ne11;

    const int tile_x_max_i = ne01 - blockIdx.x*mmq_y - 1;

    const int * y = (const int *) yc + blockIdx.y*(mmq_x*sizeof(block_q8_1_mmq)/sizeof(int));

    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE)] = {0.0f};

    for (int kb0 = 0; kb0 < blocks_per_row_x; kb0 += blocks_per_warp) {

        load_tiles(x, tile_x_qs, tile_x_dm, tile_x_sc, stride01*blockIdx.x*mmq_y + kb0, tile_x_max_i, stride01);

#pragma unroll
        for (int kr = 0; kr < qr; ++kr) {
            const int * by0 = y + stride11*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + kr*sizeof(block_q8_1_mmq)/sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k0 = kr*WARP_SIZE/qr; k0 < (kr+1)*WARP_SIZE/qr; k0 += vdr) {
                vec_dot(tile_x_qs, tile_x_dm, tile_x_sc, tile_y, sum, k0);
            }

            __syncthreads();
        }
    }

    write_back(sum, dst, ne0, ne1);
}

struct mmq_args {
    const char * x; const char * y; float * dst;
    int64_t ne00; int64_t ne01; int64_t stride01;
    int64_t ne10; int64_t ne11; int64_t stride11;
    int64_t ne0;
};

constexpr int mmq_get_nwarps(int mmq_x) {
    return mmq_x >= 32 ? 8 : 4;
}

static int mmq_get_shmem(const ggml_type type, const int mmq_x, const int mmq_y) {
    const tile_x_sizes txs = get_tile_x_sizes_host(type, mmq_y);
    const int nwarps = mmq_get_nwarps(mmq_x);

    const int shmem_x = txs.qs*sizeof(int) + txs.dm*sizeof(half2) + txs.sc*sizeof(int);
    const int shmem_y = mmq_x*WARP_SIZE*sizeof(int) + mmq_x*(WARP_SIZE/QI8_1)*sizeof(half2);
    return shmem_x + GGML_PAD(shmem_y, nwarps*WARP_SIZE*sizeof(int));
}

template <ggml_type type, int mmq_x, int nwarps>
static void launch_mul_mat_q(const mmq_args & args, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int mmq_y = get_mmq_y_host(cc, mmq_x);

    const int block_num_x = (args.ne01 + mmq_y - 1) / mmq_y;
    const int block_num_y = (args.ne11 + mmq_x - 1) / mmq_x;
    const dim3 block_nums(block_num_x, block_num_y, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    const int shmem = mmq_get_shmem(type, mmq_x, mmq_y);

#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
    static bool shmem_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shmem_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<type, mmq_x, nwarps, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q<type, mmq_x, nwarps, true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        shmem_limit_raised[id] = true;
    }
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))

    if (args.ne01 % mmq_y == 0) {
        const bool need_check = false;
        mul_mat_q<type, mmq_x, nwarps, need_check><<<block_nums, block_dims, shmem, stream>>>
            (args.x, args.y, args.dst, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
    } else {
        const bool need_check = true;
        mul_mat_q<type, mmq_x, nwarps, need_check><<<block_nums, block_dims, shmem, stream>>>
            (args.x, args.y, args.dst, args.ne00, args.ne01, args.stride01, args.ne10, args.ne11, args.stride11, args.ne0);
    }
}

template <ggml_type type>
void mul_mat_q_case(const mmq_args & args, cudaStream_t stream) {
    const int id    = ggml_cuda_get_device();
    const int nsm   = ggml_cuda_info().devices[id].nsm;
    const int cc    = ggml_cuda_info().devices[id].cc;
    const int smpbo = ggml_cuda_info().devices[id].smpbo;

    const int mmq_x_max = get_mmq_x_max_host(cc);
    const int mmq_y = get_mmq_y_host(cc, mmq_x_max);
    const int block_num_y = (args.ne01 + mmq_y - 1) / mmq_y;

    int mmq_x_best  = 0;
    int nwaves_best = INT_MAX;

    for (int mmq_x = 8; mmq_x <= mmq_x_max && nwaves_best > 1; mmq_x += 8) {
        const int block_num_x = (args.ne11 + mmq_x - 1) / mmq_x;
        const int nwaves = (block_num_x*block_num_y + nsm - 1) / nsm;

        if (nwaves < nwaves_best && mmq_get_shmem(type, mmq_x, mmq_y) <= smpbo) {
            mmq_x_best  = mmq_x;
            nwaves_best = nwaves;
        }
    }

    switch (mmq_x_best) {
        case   8:
            launch_mul_mat_q<type,   8, mmq_get_nwarps(  8)>(args, stream);
            break;
        case  16:
            launch_mul_mat_q<type,  16, mmq_get_nwarps( 16)>(args, stream);
            break;
        case  24:
            launch_mul_mat_q<type,  24, mmq_get_nwarps( 24)>(args, stream);
            break;
        case  32:
            launch_mul_mat_q<type,  32, mmq_get_nwarps( 32)>(args, stream);
            break;
        case  40:
            launch_mul_mat_q<type,  40, mmq_get_nwarps( 40)>(args, stream);
            break;
        case  48:
            launch_mul_mat_q<type,  48, mmq_get_nwarps( 48)>(args, stream);
            break;
        case  56:
            launch_mul_mat_q<type,  56, mmq_get_nwarps( 56)>(args, stream);
            break;
        case  64:
            launch_mul_mat_q<type,  64, mmq_get_nwarps( 64)>(args, stream);
            break;
        case  72:
            launch_mul_mat_q<type,  72, mmq_get_nwarps( 72)>(args, stream);
            break;
        case  80:
            launch_mul_mat_q<type,  80, mmq_get_nwarps( 80)>(args, stream);
            break;
        case  88:
            launch_mul_mat_q<type,  88, mmq_get_nwarps( 88)>(args, stream);
            break;
        case  96:
            launch_mul_mat_q<type,  96, mmq_get_nwarps( 96)>(args, stream);
            break;
        case 104:
            launch_mul_mat_q<type, 104, mmq_get_nwarps(104)>(args, stream);
            break;
        case 112:
            launch_mul_mat_q<type, 112, mmq_get_nwarps(112)>(args, stream);
            break;
        case 120:
            launch_mul_mat_q<type, 120, mmq_get_nwarps(120)>(args, stream);
            break;
        case 128:
            launch_mul_mat_q<type, 128, mmq_get_nwarps(128)>(args, stream);
            break;
        default:
            fprintf(stderr, "mmq_x_best=%d\n", mmq_x_best);
            GGML_ASSERT(false);
            break;
    }
}

#define DECL_MMQ_CASE(type)                                                        \
    template void mul_mat_q_case<type>(const mmq_args & args, cudaStream_t stream) \

extern DECL_MMQ_CASE(GGML_TYPE_Q4_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_1);
extern DECL_MMQ_CASE(GGML_TYPE_Q8_0);
extern DECL_MMQ_CASE(GGML_TYPE_Q2_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q3_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q4_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q5_K);
extern DECL_MMQ_CASE(GGML_TYPE_Q6_K);

// -------------------------------------------------------------------------------------------------------------------------

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_supports_mmq(enum ggml_type type);
