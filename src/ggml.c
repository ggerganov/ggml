// Defines CLOCK_MONOTONIC on Linux
#define _GNU_SOURCE

#include "ggml.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>

// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef static_assert
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif

#if defined(_WIN32)

#include <windows.h>

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;

static void atomic_store(atomic_int* ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int* ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int* ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_sub(atomic_int* ptr, LONG dec) {
    return atomic_fetch_add(ptr, -(dec));
}

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t* out, void* unused, thread_ret_t(*func)(void*), void* arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void* unused) {
    (void) unused;
    return (int) WaitForSingleObject(thread, INFINITE);
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else
#include <pthread.h>
#include <stdatomic.h>

typedef void* thread_ret_t;
#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#ifndef __SSE3__
#define __SSE3__
#endif
#endif

#ifdef __HAIKU__
#define static_assert(cond, msg) _Static_assert(cond, msg)
#endif

/*#define GGML_PERF*/
#define GGML_DEBUG 0
#define GGML_GELU_FP16
#define GGML_SILU_FP16

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL  2

#ifdef GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define GGML_SOFT_MAX_ACCELERATE
#endif

#if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#else
    #define GGML_MEM_ALIGN 16
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#define GGML_ALIGNED_MALLOC(size)  _aligned_malloc(size, GGML_MEM_ALIGN)
#define GGML_ALIGNED_FREE(ptr)     _aligned_free(ptr)
#else
inline static void* ggml_aligned_malloc(size_t size) {
    void* aligned_memory = NULL;
    int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
    if (result != 0) {
        // Handle allocation failure
        return NULL;
    }
    return aligned_memory;
}
#define GGML_ALIGNED_MALLOC(size)  ggml_aligned_malloc(size)
#define GGML_ALIGNED_FREE(ptr)     free(ptr)
#endif

#define UNUSED(x) (void)(x)
#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)

#define GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#elif defined(GGML_USE_CUBLAS)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "ggml-cuda.h"

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            printf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__,   \
                cudaGetErrorString(err_));                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        cublasStatus_t err_ = (err);                                           \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
            printf("cuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__);    \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static cublasHandle_t cublasH = NULL;
static cudaStream_t cudaStream = NULL;
static void init_cublas(void) {
    if (cublasH == NULL) {
        // create cublas handle, bind a stream
        CUBLAS_CHECK(cublasCreate(&cublasH));

        CUDA_CHECK(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking));

        CUBLAS_CHECK(cublasSetStream(cublasH, cudaStream));

        // configure logging to stdout
        // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, NULL));
    }
}
#endif

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// floating point type used to accumulate sums
typedef double ggml_float;

// 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
#ifdef __ARM_NEON

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>

#define GGML_COMPUTE_FP16_TO_FP32(x) ((float) (x))
#define GGML_COMPUTE_FP32_TO_FP16(x) (x)

#define GGML_FP16_TO_FP32(x) ((float) (x))
#define GGML_FP32_TO_FP16(x) (x)

#else

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#else
#ifdef __POWER9_VECTOR__
#include <altivec.h>
#undef bool
#define bool _Bool
#else
#include <immintrin.h>
#endif
#endif

#ifdef __F16C__

#ifdef _MSC_VER
#define GGML_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define GGML_COMPUTE_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#else
#define GGML_COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

#elif defined(__POWER9_VECTOR__)

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
/* the inline asm below is about 12% faster than the lookup method */
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    register float f;
    register double d;
    __asm__(
        "mtfprd %0,%2\n"
        "xscvhpdp %0,%0\n"
        "frsp %1,%0\n" :
        /* temp */ "=d"(d),
        /* out */  "=f"(f):
        /* in */   "r"(h));
    return f;
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
    register double d;
    register ggml_fp16_t r;
    __asm__( /* xscvdphp can work on double or single precision */
        "xscvdphp %0,%2\n"
        "mffprd %1,%0\n" :
        /* temp */ "=d"(d),
        /* out */  "=r"(r):
        /* in */   "f"(f));
    return r;
}

#else

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#endif // __F16C__

#endif // __ARM_NEON

//
// global data
//

// precomputed gelu table for f16 (128 KB)
static ggml_fp16_t table_gelu_f16[1 << 16];

// precomputed silu table for f16 (128 KB)
static ggml_fp16_t table_silu_f16[1 << 16];

// precomputed exp table for f16 (128 KB)
static ggml_fp16_t table_exp_f16[1 << 16];

// precomputed f32 table for f16 (256 KB)
static float table_f32_f16[1 << 16];

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into ggml_lookup_fp16_to_fp32,
// so we define GGML_FP16_TO_FP32 and GGML_FP32_TO_FP16 elsewhere for NEON.
// This is also true for POWER9.
#if !defined(GGML_FP16_TO_FP32) || !defined(GGML_FP32_TO_FP16)

inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return table_f32_f16[s];
}

#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#endif

// note: do not use these inside ggml.c
// these are meant to be used via the ggml.h API
float ggml_fp16_to_fp32(ggml_fp16_t x) {
    return (float) GGML_FP16_TO_FP32(x);
}

ggml_fp16_t ggml_fp32_to_fp16(float x) {
    return GGML_FP32_TO_FP16(x);
}

//
// timing
//

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq;
void ggml_time_init(void) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    timer_freq = frequency.QuadPart;
}
int64_t ggml_time_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (t.QuadPart * 1000) / timer_freq;
}
int64_t ggml_time_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (t.QuadPart * 1000000) / timer_freq;
}
#else
void ggml_time_init(void) {}
int64_t ggml_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t ggml_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t ggml_cycles(void) {
    return clock();
}

int64_t ggml_cycles_per_ms(void) {
    return CLOCKS_PER_SEC/1000;
}

#ifdef GGML_PERF
#define ggml_perf_time_ms()       ggml_time_ms()
#define ggml_perf_time_us()       ggml_time_us()
#define ggml_perf_cycles()        ggml_cycles()
#define ggml_perf_cycles_per_ms() ggml_cycles_per_ms()
#else
#define ggml_perf_time_ms()       0
#define ggml_perf_time_us()       0
#define ggml_perf_cycles()        0
#define ggml_perf_cycles_per_ms() 0
#endif

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);

//
// quantization
//

#if __AVX__ || __AVX2__ || __AVX512F__
// Unpack 16 4-bit fields into 16 bytes
// The output vector contains 16 bytes, each one in [ 0 .. 15 ] interval
static inline __m128i bytes_from_nibbles_16(const uint8_t * rsi)
{
    // Load 8 bytes from memory
    __m128i tmp = _mm_loadu_si64( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m128i bytes = _mm_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m128i lowMask = _mm_set1_epi8( 0xF );
    __m128i high = _mm_andnot_si128( lowMask, bytes );
    __m128i low = _mm_and_si128( lowMask, bytes );
    high = _mm_slli_epi16( high, 4 );
    bytes = _mm_or_si128( low, high );
    return bytes;
}

#if __AVX2__ || __AVX512F__
// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmp = _mm_loadu_si128( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m256i bytes = _mm256_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    __m256i high = _mm256_andnot_si256( lowMask, bytes );
    __m256i low = _mm256_and_si256( lowMask, bytes );
    high = _mm256_slli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );
    return bytes;
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
}
#else
static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}
#endif
#endif // __AVX__ || __AVX2__ || __AVX512F__

#if __ARM_NEON

#if !defined(__aarch64__)

inline static uint16_t vaddvq_u8(uint8x16_t v) {
    return
        (uint16_t)vgetq_lane_u8(v, 0)  + (uint16_t)vgetq_lane_u8(v, 1)  +
        (uint16_t)vgetq_lane_u8(v, 2)  + (uint16_t)vgetq_lane_u8(v, 3)  +
        (uint16_t)vgetq_lane_u8(v, 4)  + (uint16_t)vgetq_lane_u8(v, 5)  +
        (uint16_t)vgetq_lane_u8(v, 6)  + (uint16_t)vgetq_lane_u8(v, 7)  +
        (uint16_t)vgetq_lane_u8(v, 8)  + (uint16_t)vgetq_lane_u8(v, 9)  +
        (uint16_t)vgetq_lane_u8(v, 10) + (uint16_t)vgetq_lane_u8(v, 11) +
        (uint16_t)vgetq_lane_u8(v, 12) + (uint16_t)vgetq_lane_u8(v, 13) +
        (uint16_t)vgetq_lane_u8(v, 14) + (uint16_t)vgetq_lane_u8(v, 15);
}

inline static int16_t vaddvq_s8(int8x16_t v) {
    return
        (int16_t)vgetq_lane_s8(v, 0)  + (int16_t)vgetq_lane_s8(v, 1)  +
        (int16_t)vgetq_lane_s8(v, 2)  + (int16_t)vgetq_lane_s8(v, 3)  +
        (int16_t)vgetq_lane_s8(v, 4)  + (int16_t)vgetq_lane_s8(v, 5)  +
        (int16_t)vgetq_lane_s8(v, 6)  + (int16_t)vgetq_lane_s8(v, 7)  +
        (int16_t)vgetq_lane_s8(v, 8)  + (int16_t)vgetq_lane_s8(v, 9)  +
        (int16_t)vgetq_lane_s8(v, 10) + (int16_t)vgetq_lane_s8(v, 11) +
        (int16_t)vgetq_lane_s8(v, 12) + (int16_t)vgetq_lane_s8(v, 13) +
        (int16_t)vgetq_lane_s8(v, 14) + (int16_t)vgetq_lane_s8(v, 15);
}

inline static int32_t vaddvq_s16(int16x8_t v) {
    return
        (int32_t)vgetq_lane_s16(v, 0) + (int32_t)vgetq_lane_s16(v, 1) +
        (int32_t)vgetq_lane_s16(v, 2) + (int32_t)vgetq_lane_s16(v, 3) +
        (int32_t)vgetq_lane_s16(v, 4) + (int32_t)vgetq_lane_s16(v, 5) +
        (int32_t)vgetq_lane_s16(v, 6) + (int32_t)vgetq_lane_s16(v, 7);
}

inline static uint32_t vaddvq_u16(uint16x8_t v) {
    return
        (uint32_t)vgetq_lane_u16(v, 0) + (uint32_t)vgetq_lane_u16(v, 1) +
        (uint32_t)vgetq_lane_u16(v, 2) + (uint32_t)vgetq_lane_u16(v, 3) +
        (uint32_t)vgetq_lane_u16(v, 4) + (uint32_t)vgetq_lane_u16(v, 5) +
        (uint32_t)vgetq_lane_u16(v, 6) + (uint32_t)vgetq_lane_u16(v, 7);
}

inline static int32_t vaddvq_s32(int32x4_t v) {
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

inline static float vaddvq_f32(float32x4_t v) {
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

float vminvq_f32(float32x4_t v) {
    return
        MIN(MIN(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MIN(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

float vmaxvq_f32(float32x4_t v) {
    return
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

int8x8_t vzip1_s8(int8x8_t a, int8x8_t b) {
    return vget_low_s8(vcombine_s8(a, b));
}

int8x8_t vzip2_s8(int8x8_t a, int8x8_t b) {
    return vget_high_s8(vcombine_s8(a, b));
}

uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    return vget_low_u8(vcombine_u8(a, b));
}

uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    return vget_high_u8(vcombine_u8(a, b));
}

#endif
#endif


#define QK4_0 32
typedef struct {
    float   d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(float) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    float   d;          // delta
    float   m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(float) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK4_2 16
typedef struct {
    ggml_fp16_t d;         // delta
    uint8_t qs[QK4_2 / 2]; // nibbles / quants
} block_q4_2;
static_assert(sizeof(block_q4_2) == sizeof(ggml_fp16_t) + QK4_2 / 2, "wrong q4_2 block size/padding");

#define QK4_3 16
typedef struct {
    ggml_fp16_t d;         // delta
    ggml_fp16_t m;         // min
    uint8_t qs[QK4_3 / 2]; // nibbles / quants
} block_q4_3;
static_assert(sizeof(block_q4_3) == 2 * sizeof(ggml_fp16_t) + QK4_3 / 2, "wrong q4_3 block size/padding");

#define QK8_0 32
typedef struct {
    float   d;          // delta
    int8_t  qs[QK8_0];  // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(float) + QK8_0, "wrong q8_0 block size/padding");


// reference implementation for deterministic creation of model files
static void quantize_row_q4_0_reference(const float * restrict x, block_q4_0 * restrict y, int k) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    uint8_t pp[QK4_0/2];

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int l = 0; l < QK4_0; l++) {
            const float v = x[i*QK4_0 + l];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK4_0; l += 2) {
            const float v0 = x[i*QK4_0 + l + 0]*id;
            const float v1 = x[i*QK4_0 + l + 1]*id;

            const uint8_t vi0 = (int8_t)roundf(v0) + 8;
            const uint8_t vi1 = (int8_t)roundf(v1) + 8;

            assert(vi0 < 16);
            assert(vi1 < 16);

            pp[l/2] = vi0 | (vi1 << 4);
        }

        memcpy(y[i].qs, pp, sizeof(pp));
    }
}

static void quantize_row_q4_0(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    block_q4_0 * restrict y = vy;

#if defined(__POWER9_VECTOR__)
    const vector float v85 = vec_splats(8.5f);
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        vector float srcv [8];
        vector float asrcv[8];
        vector float amaxv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = *(vector float *)(x + i*32 + 4*l);
        for (int l = 0; l < 8; l++) asrcv[l] = vec_abs(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2*l] = vec_max(asrcv[2*l], asrcv[2*l+1]);
        //for (int l = 0; l < 2; l++) amaxv[4*l] = vec_max(amaxv[4*l], amaxv[4*l+2]);
        amaxv[0] = vec_max(amaxv[0], amaxv[2]);
        amaxv[4] = vec_max(amaxv[4], amaxv[6]);
        //for (int l = 0; l < 1; l++) amaxv[8*l] = vec_max(amaxv[8*l], amaxv[8*l+4]);
        amaxv[0] = vec_max(amaxv[0], amaxv[4]);

        amax = MAX(
                MAX(vec_extract(amaxv[0], 0), vec_extract(amaxv[0], 1)),
                MAX(vec_extract(amaxv[0], 2), vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0/d : 0.0;

        y[i].d = d;

        const vector float vid = vec_splats(id);
        uint8_t * restrict pb = y[i].qs;
        for (int l = 0; l < 8; l++) {
            const vector float vf  = vec_madd(srcv[l], vid, v85);
            const vector signed int vi = vec_signed(vf);

            pb[2*l + 0] = vec_extract(vi, 0) | (vec_extract(vi, 1) << 4);
            pb[2*l + 1] = vec_extract(vi, 2) | (vec_extract(vi, 3) << 4);
        }
    }
#elif __ARM_NEON
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = vld1q_f32(x + i*32 + 4*l);
        for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2*l] = vmaxq_f32(asrcv[2*l], asrcv[2*l+1]);
        for (int l = 0; l < 2; l++) amaxv[4*l] = vmaxq_f32(amaxv[4*l], amaxv[4*l+2]);
        for (int l = 0; l < 1; l++) amaxv[8*l] = vmaxq_f32(amaxv[8*l], amaxv[8*l+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < 8; l++) {
            const float32x4_t v  = vmulq_n_f32(srcv[l], id);
            const float32x4_t vf = vaddq_f32(v, vdupq_n_f32(8.5f));
            const int32x4_t   vi = vcvtq_s32_f32(vf);

            y[i].qs[2*l + 0] = vgetq_lane_s32(vi, 0) | (vgetq_lane_s32(vi, 1) << 4);
            y[i].qs[2*l + 1] = vgetq_lane_s32(vi, 2) | (vgetq_lane_s32(vi, 3) << 4);
        }
    }
#elif defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 7.0f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 7.0f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        // Apply offset to translate the range from [ -7 .. +7 ] into [ +1 .. +15 ]
        const __m256i off = _mm256_set1_epi8( 8 );
        i0 = _mm256_add_epi8( i0, off );

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles( i0 );
        _mm_storeu_si128( ( __m128i* )y[i].qs, res );
    }
#elif defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 7.0f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 7.0f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        // Apply offset to translate the range from [ -7 .. +7 ] into [ +1 .. +15 ]
        const __m128i off = _mm_set1_epi8( 8);
        ni0 = _mm_add_epi8( ni0, off );
        ni4 = _mm_add_epi8( ni4, off );

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles( ni0, ni4 );
        _mm_storeu_si128( ( __m128i* )y[i].qs, res );
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = wasm_v128_load(x + i*32 + 4*l);
        for (int l = 0; l < 8; l++) asrcv[l] = wasm_f32x4_abs(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2*l] = wasm_f32x4_max(asrcv[2*l], asrcv[2*l+1]);
        for (int l = 0; l < 2; l++) amaxv[4*l] = wasm_f32x4_max(amaxv[4*l], amaxv[4*l+2]);
        for (int l = 0; l < 1; l++) amaxv[8*l] = wasm_f32x4_max(amaxv[8*l], amaxv[8*l+4]);

        amax = MAX(
                MAX(wasm_f32x4_extract_lane(amaxv[0], 0), wasm_f32x4_extract_lane(amaxv[0], 1)),
                MAX(wasm_f32x4_extract_lane(amaxv[0], 2), wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 3) - 1);
        const float id = d ? 1.0/d : 0.0;

        y[i].d = d;

        for (int l = 0; l < 8; l++) {
            const v128_t v  = wasm_f32x4_mul(srcv[l], wasm_f32x4_splat(id));
            const v128_t vf = wasm_f32x4_add(v, wasm_f32x4_splat(8.5f));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(vf);

            y[i].qs[2*l + 0] = wasm_i32x4_extract_lane(vi, 0) | (wasm_i32x4_extract_lane(vi, 1) << 4);
            y[i].qs[2*l + 1] = wasm_i32x4_extract_lane(vi, 2) | (wasm_i32x4_extract_lane(vi, 3) << 4);
        }
    }
#else
    // scalar
    quantize_row_q4_0_reference(x, y, k);
#endif
}

static void quantize_row_q4_1_reference(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_1 == 0);
    const int nb = k / QK4_1;

    block_q4_1 * restrict y = vy;

    uint8_t pp[QK4_1/2];

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int l = 0; l < QK4_1; l++) {
            const float v = x[i*QK4_1 + l];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;
        y[i].m = min;

        for (int l = 0; l < QK4_1; l += 2) {
            const float v0 = (x[i*QK4_1 + l + 0] - min)*id;
            const float v1 = (x[i*QK4_1 + l + 1] - min)*id;

            const uint8_t vi0 = roundf(v0);
            const uint8_t vi1 = roundf(v1);

            assert(vi0 < 16);
            assert(vi1 < 16);

            pp[l/2] = vi0 | (vi1 << 4);
        }

        memcpy(y[i].qs, pp, sizeof(pp));
    }
}

static void quantize_row_q4_1(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_1 == 0);

    const int nb = k / QK4_1;

    block_q4_1 * restrict y = vy;

#if defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max for the block
        __m256 vmax;
        vmax = _mm256_max_ps( v0, v1 );
        vmax = _mm256_max_ps( vmax, v2 );
        vmax = _mm256_max_ps( vmax, v3 );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( vmax, 1 ), _mm256_castps256_ps128( vmax ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Compute min for the block
        __m256 vmin;
        vmin = _mm256_min_ps( v0, v1 );
        vmin = _mm256_min_ps( vmin, v2 );
        vmin = _mm256_min_ps( vmin, v3 );

        __m128 min4 = _mm_min_ps( _mm256_extractf128_ps( vmin, 1 ), _mm256_castps256_ps128( vmin ) );
        min4 = _mm_min_ps( min4, _mm_movehl_ps( min4, min4 ) );
        min4 = _mm_min_ss( min4, _mm_movehdup_ps( min4 ) );
        const float minScalar = _mm_cvtss_f32( min4 );

        // Quantize these floats
        const float d = (maxScalar - minScalar) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].m = minScalar;
        y[i].d = d;

        // x = (x-min)*id
        const __m256 mul = _mm256_set1_ps( id );
        const __m256 off = _mm256_set1_ps( minScalar );
        v0 = _mm256_mul_ps( _mm256_sub_ps( v0, off ), mul );
        v1 = _mm256_mul_ps( _mm256_sub_ps( v1, off ), mul );
        v2 = _mm256_mul_ps( _mm256_sub_ps( v2, off ), mul );
        v3 = _mm256_mul_ps( _mm256_sub_ps( v3, off ), mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles( i0 );
        _mm_storeu_si128( ( __m128i* )y[i].qs, res );
    }
#elif __ARM_NEON
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv[8];
        float32x4_t minv[8];
        float32x4_t maxv[8];

        for (int l = 0; l < 8; l++) srcv[l] = vld1q_f32(x + i*QK4_1 + 4*l);

        for (int l = 0; l < 4; l++) minv[2*l] = vminq_f32(srcv[2*l], srcv[2*l + 1]);
        for (int l = 0; l < 2; l++) minv[4*l] = vminq_f32(minv[4*l], minv[4*l + 2]);
        for (int l = 0; l < 1; l++) minv[8*l] = vminq_f32(minv[8*l], minv[8*l + 4]);

        for (int l = 0; l < 4; l++) maxv[2*l] = vmaxq_f32(srcv[2*l], srcv[2*l + 1]);
        for (int l = 0; l < 2; l++) maxv[4*l] = vmaxq_f32(maxv[4*l], maxv[4*l + 2]);
        for (int l = 0; l < 1; l++) maxv[8*l] = vmaxq_f32(maxv[8*l], maxv[8*l + 4]);

        const float min = vminvq_f32(minv[0]);
        const float max = vmaxvq_f32(maxv[0]);

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;
        y[i].m = min;

        const float32x4_t minv0 = vdupq_n_f32(min);

        for (int l = 0; l < 8; l++) {
            const float32x4_t v  = vmulq_n_f32(vsubq_f32(srcv[l], minv0), id);
            const float32x4_t vf = vaddq_f32(v, vdupq_n_f32(0.5f)); // needed to round to nearest
            const int32x4_t   vi = vcvtq_s32_f32(vf);

            y[i].qs[2*l + 0] = vgetq_lane_s32(vi, 0) | (vgetq_lane_s32(vi, 1) << 4);
            y[i].qs[2*l + 1] = vgetq_lane_s32(vi, 2) | (vgetq_lane_s32(vi, 3) << 4);
        }
    }
#else
    // scalar
    quantize_row_q4_1_reference(x, vy, k);
#endif
}

// reference implementation for deterministic creation of model files
static void quantize_row_q4_2_reference(const float * restrict x, block_q4_2 * restrict y, int k) {
    assert(k % QK4_2 == 0);

    const int nb = k / QK4_2;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int l = 0; l < QK4_2; l++) {
            const float v = x[i*QK4_2 + l];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 3) - 1);

        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int l = 0; l < QK4_2; l += 2) {
            const float v0 = x[i*QK4_2 + l + 0]*id;
            const float v1 = x[i*QK4_2 + l + 1]*id;

            const uint8_t vi0 = (uint8_t)(v0 + 8.5f);
            const uint8_t vi1 = (uint8_t)(v1 + 8.5f);

            assert(vi0 < 16);
            assert(vi1 < 16);

            y[i].qs[l/2] = vi0 | (vi1 << 4);
        }
    }
}

static inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float kquantize_q4_with_bounds(int n, int nmin, int nmax, const float * restrict X, int nCandidates,
        const float * restrict candidates, int8_t * restrict L) {
    assert (nmin >= INT8_MIN);
    assert (nmax <= INT8_MAX);
    float amax = 0;
    for (int i=0; i<n; ++i) amax = MAX(amax, fabsf(X[i]));
    if (!amax) { // all zero
        for (int i=0; i<n; ++i) L[i] = 0;
        return 1.f;
    }
    float best = 0, bestScale = 0;
    for (int si=0; si<nCandidates; ++si) {
        float iscale = candidates[si]/amax;
        float sumlxP = 0; int suml2P = 0;
        float sumlxM = 0; int suml2M = 0;
        for (int i=0; i<n; ++i) {
            int l = nearest_int(iscale*X[i]);
            int lp = MAX(nmin, MIN(nmax, +l));
            int lm = MAX(nmin, MIN(nmax, -l));
            sumlxP += X[i]*lp; suml2P += lp*lp;
            sumlxM += X[i]*lm; suml2M += lm*lm;
        }
        float sumlxP2 = sumlxP*sumlxP;
        float sumlxM2 = sumlxM*sumlxM;
        if (sumlxP2*suml2M > sumlxM2*suml2P) {
            if (sumlxP2 > best*suml2P) {
                best = sumlxP2/suml2P; bestScale = iscale;
            }
        } else {
            if (sumlxM2 > best*suml2M) {
                best = sumlxM2/suml2M; bestScale = -iscale;
            }
        }
    }
    float sumlx = 0; int suml2 = 0;
    for (int i=0; i<n; ++i) {
        int l = nearest_int(bestScale*X[i]);
        l = MAX(nmin, MIN(nmax, l));
        sumlx += X[i]*l; suml2 += l*l;
        L[i] = l;
    }
    float scale = sumlx/suml2;
    return scale;
}

static void quantize_row_q4_2_rmse(const float * restrict x, block_q4_2 * restrict y, int k) {
#define CANDIDATE_COUNT 8
    static const float candidates[CANDIDATE_COUNT] = { +8.7f, +8.3f, +8.1f, +7.8f, +7.3f, +7.0f, +6.3f, +5.7f };
    assert(k % QK4_2 == 0);

    int8_t L[QK4_2];

    const int nb = k / QK4_2;

    for (int i = 0; i < nb; i++) {
        float scale = kquantize_q4_with_bounds(QK4_2, -8, 7, x, CANDIDATE_COUNT, candidates, L);
        y[i].d = GGML_FP32_TO_FP16(scale);

        for (int l = 0; l < QK4_2; l += 2) {
            const uint8_t vi0 = (uint8_t)(L[l+0] + 8);
            const uint8_t vi1 = (uint8_t)(L[l+1] + 8);

            assert(vi0 < 16);
            assert(vi1 < 16);

            y[i].qs[l/2] = vi0 | (vi1 << 4);
        }

        x += QK4_2;
    }
}

static void quantize_row_q4_2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_2 == 0);

    block_q4_2 * restrict y = vy;

    //quantize_row_q4_2_reference(x, y, k);
    // This produces the exact same format, just better match to the input floats ("better" as measured by RMSE)
    quantize_row_q4_2_rmse(x, y, k);
}

static void quantize_row_q4_3_reference(const float * restrict x, block_q4_3 * restrict y, int k) {
    assert(k % QK4_3 == 0);
    const int nb = k / QK4_3;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int l = 0; l < QK4_3; l++) {
            const float v = x[i*QK4_3 + l];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        for (int l = 0; l < QK4_3; l += 2) {
            const float v0 = (x[i*QK4_3 + l + 0] - min)*id;
            const float v1 = (x[i*QK4_3 + l + 1] - min)*id;

            const uint8_t vi0 = (int) (v0 + 0.5f);
            const uint8_t vi1 = (int) (v1 + 0.5f);

            assert(vi0 < 16);
            assert(vi1 < 16);

            y[i].qs[l/2] = vi0 | (vi1 << 4);
        }
    }
}

static void quantize_row_q4_3(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_3 == 0);

    block_q4_3 * restrict y = vy;

    quantize_row_q4_3_reference(x, y, k);
}

// reference implementation for deterministic creation of model files
static void quantize_row_q8_0_reference(const float * restrict x, block_q8_0 * restrict y, int k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int l = 0; l < QK8_0; l++) {
            const float v = x[i*QK8_0 + l];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK8_0; ++l) {
            const float v = x[i*QK8_0 + l]*id;
            y[i].qs[l] = roundf(v);
        }
    }
}

static void quantize_row_q8_0(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = vld1q_f32(x + i*32 + 4*l);
        for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2*l] = vmaxq_f32(asrcv[2*l], asrcv[2*l+1]);
        for (int l = 0; l < 2; l++) amaxv[4*l] = vmaxq_f32(amaxv[4*l], amaxv[4*l+2]);
        for (int l = 0; l < 1; l++) amaxv[8*l] = vmaxq_f32(amaxv[8*l], amaxv[8*l+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < 8; l++) {
            const float32x4_t v  = vmulq_n_f32(srcv[l], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*l + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*l + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*l + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*l + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#else
    // scalar
    quantize_row_q8_0_reference(x, y, k);
#endif
}

static void dequantize_row_q4_0(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    const block_q4_0 * restrict x = vx;

#if defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        // scale factor
        const __m256 d_v = _mm256_broadcast_ss(&x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 32) {
            // Load 32x4-bit integers into 32x8-bit integers
            __m256i vx8 = bytes_from_nibbles_32(pp+l/2);

            // Subtract 8 from the integers
            vx8 = _mm256_sub_epi8(vx8, _mm256_set1_epi8(8));

            // Convert to 16-bit int
            const __m256i vx16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 0));
            const __m256i vx16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 1));

            // Convert to 32-bit int -> float 32
            const __m256 vf[4] = {
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 1))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 1)))
            };

            // Scale and store
            for (int j = 0; j < 4; j++) {
                const __m256 result = _mm256_mul_ps(vf[j], d_v);
                _mm256_storeu_ps(y + i * QK4_0 + l + j*8, result);
            }
        }
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        const float32x4_t vd = vdupq_n_f32(x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 16) {
            // Load 16x4-bit integers into 8x8-bit integers
            const uint8x8_t v8 = vld1_u8(pp + l/2);

            // Expand 4-bit qs to 8-bit bytes
            const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0f));
            const uint8x8_t v1 = vshr_n_u8(v8, 4);

            // Convert to signed 8-bit integers
            const int8x8_t vs_0 = vreinterpret_s8_u8(v0);
            const int8x8_t vs_1 = vreinterpret_s8_u8(v1);

            // Subtract 8 from each byte
            const int8x8_t vb_0 = vsub_s8(vs_0, vdup_n_s8(8));
            const int8x8_t vb_1 = vsub_s8(vs_1, vdup_n_s8(8));

            // Interleave and combine
            const int8x8_t vx_0 = vzip1_s8(vb_0, vb_1);
            const int8x8_t vx_1 = vzip2_s8(vb_0, vb_1);

            const int8x16_t vq = vcombine_s8(vx_0, vx_1);

            // convert to 2x int16x8_t
            const int16x8_t vi_0 = vmovl_s8(vget_low_s8 (vq));
            const int16x8_t vi_1 = vmovl_s8(vget_high_s8(vq));

            // convert to 4x float32x4_t
            const float32x4_t vf_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vi_0)));
            const float32x4_t vf_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_0)));
            const float32x4_t vf_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vi_1)));
            const float32x4_t vf_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_1)));

            // Multiply by d
            const float32x4_t r0 = vmulq_f32(vf_0, vd);
            const float32x4_t r1 = vmulq_f32(vf_1, vd);
            const float32x4_t r2 = vmulq_f32(vf_2, vd);
            const float32x4_t r3 = vmulq_f32(vf_3, vd);

            // Store
            vst1q_f32(y + i*QK4_0 + l +  0, r0);
            vst1q_f32(y + i*QK4_0 + l +  4, r1);
            vst1q_f32(y + i*QK4_0 + l +  8, r2);
            vst1q_f32(y + i*QK4_0 + l + 12, r3);
        }
    }
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8)*d;
            const float v1 = (vi1 - 8)*d;

            //printf("d = %f, vi = %d, vi0 = %d, vi1 = %d, v0 = %f, v1 = %f\n", d, vi, vi0, vi1, v0, v1);

            y[i*QK4_0 + l + 0] = v0;
            y[i*QK4_0 + l + 1] = v1;

            assert(!isnan(y[i*QK4_0 + l + 0]));
            assert(!isnan(y[i*QK4_0 + l + 1]));
        }
    }
#endif
}

static void dequantize_row_q4_1(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_1 == 0);
    const int nb = k / QK4_1;

    const block_q4_1 * restrict x = vx;

#if defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        const __m256 d_v = _mm256_broadcast_ss(&x[i].d);
        const __m256 d_m = _mm256_broadcast_ss(&x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 32) {
            // Load 32x4-bit integers into 32x8-bit integers
            __m256i vx8 = bytes_from_nibbles_32(pp+l/2);

            // Convert to 16-bit int
            const __m256i vx16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 0));
            const __m256i vx16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 1));

            // Convert to 32-bit int -> float 32
            const __m256 vf[4] = {
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 1))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 1)))
            };

            // Scale, add m and store
            for (int j = 0; j < 4; j++) {
                const __m256 result = _mm256_add_ps(_mm256_mul_ps(vf[j], d_v), d_m);
                _mm256_storeu_ps(y + i * QK4_1 + l + j*8, result);
            }
        }
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        const float32x4_t vd = vdupq_n_f32(x[i].d);
        const float32x4_t vm = vdupq_n_f32(x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 16) {
            // Load 16x4-bit integers into 8x8-bit integers
            const uint8x8_t v8 = vld1_u8(pp + l/2);

            // Expand 4-bit qs to 8-bit bytes
            const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0f));
            const uint8x8_t v1 = vshr_n_u8(v8, 4);

            // Interleave and combine
            const uint8x8_t vx_0 = vzip1_u8(v0, v1);
            const uint8x8_t vx_1 = vzip2_u8(v0, v1);

            const uint8x16_t vq = vcombine_u8(vx_0, vx_1);

            // convert to 2x uint16x8_t
            const uint16x8_t vi_0 = vmovl_u8(vget_low_u8 (vq));
            const uint16x8_t vi_1 = vmovl_u8(vget_high_u8(vq));

            // convert to 4x float32x4_t
            const float32x4_t vf_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (vi_0)));
            const float32x4_t vf_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vi_0)));
            const float32x4_t vf_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (vi_1)));
            const float32x4_t vf_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vi_1)));

            // multiply by d and add m
            const float32x4_t r0 = vmlaq_f32(vm, vf_0, vd);
            const float32x4_t r1 = vmlaq_f32(vm, vf_1, vd);
            const float32x4_t r2 = vmlaq_f32(vm, vf_2, vd);
            const float32x4_t r3 = vmlaq_f32(vm, vf_3, vd);

            // Store
            vst1q_f32(y + i*QK4_1 + l +  0, r0);
            vst1q_f32(y + i*QK4_1 + l +  4, r1);
            vst1q_f32(y + i*QK4_1 + l +  8, r2);
            vst1q_f32(y + i*QK4_1 + l + 12, r3);
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;
        const float m = x[i].m;

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = vi0*d + m;
            const float v1 = vi1*d + m;

            y[i*QK4_1 + l + 0] = v0;
            y[i*QK4_1 + l + 1] = v1;

            assert(!isnan(y[i*QK4_1 + l + 0]));
            assert(!isnan(y[i*QK4_1 + l + 1]));
        }
    }
#endif
}

static void dequantize_row_q4_2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_2 == 0);
    const int nb = k / QK4_2;

    const block_q4_2 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_2; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8)*d;
            const float v1 = (vi1 - 8)*d;

            y[i*QK4_2 + l + 0] = v0;
            y[i*QK4_2 + l + 1] = v1;

            assert(!isnan(y[i*QK4_2 + l + 0]));
            assert(!isnan(y[i*QK4_2 + l + 1]));
        }
    }
}

static void dequantize_row_q4_3(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_3 == 0);
    const int nb = k / QK4_3;

    const block_q4_3 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_3; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0xf;
            const int8_t vi1 = vi >> 4;

            const float v0 = vi0*d + m;
            const float v1 = vi1*d + m;

            y[i*QK4_3 + l + 0] = v0;
            y[i*QK4_3 + l + 1] = v1;

            assert(!isnan(y[i*QK4_3 + l + 0]));
            assert(!isnan(y[i*QK4_3 + l + 1]));
        }
    }
}

static void ggml_vec_dot_q4_0_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q4_1_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q4_2_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q4_3_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);

static const quantize_fns_t quantize_fns[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0] = {
        .dequantize_row_q         = dequantize_row_q4_0,
        .quantize_row_q           = quantize_row_q4_0,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q4_0_reference,
        .quantize_row_q_dot       = quantize_row_q8_0,
        .vec_dot_q                = ggml_vec_dot_q4_0_q8_0,
    },
    [GGML_TYPE_Q4_1] = {
        .dequantize_row_q         = dequantize_row_q4_1,
        .quantize_row_q           = quantize_row_q4_1,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q4_1_reference,
        .quantize_row_q_dot       = quantize_row_q8_0,
        .vec_dot_q                = ggml_vec_dot_q4_1_q8_0,
    },
    [GGML_TYPE_Q4_2] = {
        .dequantize_row_q         = dequantize_row_q4_2,
        .quantize_row_q           = quantize_row_q4_2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q4_2_rmse, //quantize_row_q4_2_reference,
        .quantize_row_q_dot       = quantize_row_q8_0,
        .vec_dot_q                = ggml_vec_dot_q4_2_q8_0,
    },
    [GGML_TYPE_Q4_3] = {
        .dequantize_row_q         = dequantize_row_q4_3,
        .quantize_row_q           = quantize_row_q4_3,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q4_3_reference, // TODO: RMSE optimization
        .quantize_row_q_dot       = quantize_row_q8_0,
        .vec_dot_q                = ggml_vec_dot_q4_3_q8_0,
    },
    [GGML_TYPE_Q8_0] = {
        .dequantize_row_q         = NULL,   // TODO
        .quantize_row_q           = quantize_row_q8_0,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q8_0_reference,
        .quantize_row_q_dot       = quantize_row_q8_0,
        .vec_dot_q                = NULL,   // TODO
    },
};

// For internal test use
quantize_fns_t ggml_internal_get_quantize_fn(size_t i) {
    GGML_ASSERT(i < GGML_TYPE_COUNT);
    return quantize_fns[i];
}


//
// simd mappings
//

// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// GGML_F32_STEP / GGML_F16_STEP
//   number of elements to process in a single step
//
// GGML_F32_EPR / GGML_F16_EPR
//   number of elements to fit in a single register
//

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

#define GGML_SIMD

// F32 NEON

#define GGML_F32_STEP 16
#define GGML_F32_EPR  4

#define GGML_F32x4              float32x4_t
#define GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#define GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#define GGML_F32x4_LOAD         vld1q_f32
#define GGML_F32x4_STORE        vst1q_f32
#define GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define GGML_F32x4_ADD          vaddq_f32
#define GGML_F32x4_MUL          vmulq_f32
#define GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#define GGML_F32x4_REDUCE(res, x)              \
{                                              \
    for (int i = 0; i < GGML_F32_ARR/2; ++i) { \
        x[2*i] = vaddq_f32(x[2*i], x[2*i+1]);  \
    }                                          \
    for (int i = 0; i < GGML_F32_ARR/4; ++i) { \
        x[4*i] = vaddq_f32(x[4*i], x[4*i+2]);  \
    }                                          \
    for (int i = 0; i < GGML_F32_ARR/8; ++i) { \
        x[8*i] = vaddq_f32(x[8*i], x[8*i+4]);  \
    }                                          \
    res = GGML_F32x4_REDUCE_ONE(x[0]);         \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define GGML_F16_STEP 32
    #define GGML_F16_EPR  8

    #define GGML_F16x8              float16x8_t
    #define GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #define GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #define GGML_F16x8_LOAD         vld1q_f16
    #define GGML_F16x8_STORE        vst1q_f16
    #define GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #define GGML_F16x8_ADD          vaddq_f16
    #define GGML_F16x8_MUL          vmulq_f16
    #define GGML_F16x8_REDUCE(res, x)                             \
    {                                                             \
        for (int i = 0; i < GGML_F16_ARR/2; ++i) {                \
            x[2*i] = vaddq_f16(x[2*i], x[2*i+1]);                 \
        }                                                         \
        for (int i = 0; i < GGML_F16_ARR/4; ++i) {                \
            x[4*i] = vaddq_f16(x[4*i], x[4*i+2]);                 \
        }                                                         \
        for (int i = 0; i < GGML_F16_ARR/8; ++i) {                \
            x[8*i] = vaddq_f16(x[8*i], x[8*i+4]);                 \
        }                                                         \
        const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 (x[0])); \
        const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(x[0])); \
        res = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    }

    #define GGML_F16_VEC                GGML_F16x8
    #define GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #define GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #define GGML_F16_VEC_LOAD(p, i)     GGML_F16x8_LOAD(p)
    #define GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE(p, r[i])
    #define GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #define GGML_F16_VEC_ADD            GGML_F16x8_ADD
    #define GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #define GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#else
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #define GGML_F16_STEP 16
    #define GGML_F16_EPR  4

    #define GGML_F32Cx4              float32x4_t
    #define GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #define GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #define GGML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16(x))
    #define GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #define GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #define GGML_F32Cx4_ADD          vaddq_f32
    #define GGML_F32Cx4_MUL          vmulq_f32
    #define GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #define GGML_F16_VEC                GGML_F32Cx4
    #define GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #define GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx4_LOAD(p)
    #define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE(p, r[i])
    #define GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #define GGML_F16_VEC_ADD            GGML_F32Cx4_ADD
    #define GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #define GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#endif

#elif defined(__AVX__)

#define GGML_SIMD

// F32 AVX

#define GGML_F32_STEP 32
#define GGML_F32_EPR  8

#define GGML_F32x8         __m256
#define GGML_F32x8_ZERO    _mm256_setzero_ps()
#define GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#define GGML_F32x8_LOAD    _mm256_loadu_ps
#define GGML_F32x8_STORE   _mm256_storeu_ps
#if defined(__FMA__)
    #define GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
    #define GGML_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define GGML_F32x8_ADD     _mm256_add_ps
#define GGML_F32x8_MUL     _mm256_mul_ps
#define GGML_F32x8_REDUCE(res, x)                                 \
{                                                                 \
    for (int i = 0; i < GGML_F32_ARR/2; ++i) {                    \
        x[2*i] = _mm256_add_ps(x[2*i], x[2*i+1]);                 \
    }                                                             \
    for (int i = 0; i < GGML_F32_ARR/4; ++i) {                    \
        x[4*i] = _mm256_add_ps(x[4*i], x[4*i+2]);                 \
    }                                                             \
    for (int i = 0; i < GGML_F32_ARR/8; ++i) {                    \
        x[8*i] = _mm256_add_ps(x[8*i], x[8*i+4]);                 \
    }                                                             \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));                     \
}
// TODO: is this optimal ?

#define GGML_F32_VEC        GGML_F32x8
#define GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x8_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x8_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x8_STORE
#define GGML_F32_VEC_FMA    GGML_F32x8_FMA
#define GGML_F32_VEC_ADD    GGML_F32x8_ADD
#define GGML_F32_VEC_MUL    GGML_F32x8_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#define GGML_F16_STEP 32
#define GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define GGML_F32Cx8             __m256
#define GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#define GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x)))
#define GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#else
static inline __m256 __avx_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++)
        tmp[i] = GGML_FP16_TO_FP32(x[i]);

    return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    float arr[8];

    _mm256_storeu_ps(arr, y);

    for (int i = 0; i < 8; i++)
        x[i] = GGML_FP32_TO_FP16(arr[i]);
}
#define GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#endif

#define GGML_F32Cx8_FMA         GGML_F32x8_FMA
#define GGML_F32Cx8_ADD         _mm256_add_ps
#define GGML_F32Cx8_MUL         _mm256_mul_ps
#define GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#define GGML_F16_VEC                GGML_F32Cx8
#define GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#define GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx8_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#define GGML_F16_VEC_ADD            GGML_F32Cx8_ADD
#define GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#define GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#elif defined(__POWER9_VECTOR__)

#define GGML_SIMD

// F32 POWER9

#define GGML_F32_STEP 32
#define GGML_F32_EPR  4

#define GGML_F32x4              vector float
#define GGML_F32x4_ZERO         0.0f
#define GGML_F32x4_SET1         vec_splats
#define GGML_F32x4_LOAD(p)      vec_xl(0, p)
#define GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#define GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#define GGML_F32x4_ADD          vec_add
#define GGML_F32x4_MUL          vec_mul
#define GGML_F32x4_REDUCE(res, x)              \
{                                              \
    for (int i = 0; i < GGML_F32_ARR/2; ++i) { \
        x[2*i] = vec_add(x[2*i], x[2*i+1]);    \
    }                                          \
    for (int i = 0; i < GGML_F32_ARR/4; ++i) { \
        x[4*i] = vec_add(x[4*i], x[4*i+2]);    \
    }                                          \
    for (int i = 0; i < GGML_F32_ARR/8; ++i) { \
        x[8*i] = vec_add(x[8*i], x[8*i+4]);    \
    }                                          \
    res = vec_extract(x[0], 0) +               \
          vec_extract(x[0], 1) +               \
          vec_extract(x[0], 2) +               \
          vec_extract(x[0], 3);                \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#define GGML_F16_STEP       GGML_F32_STEP
#define GGML_F16_EPR        GGML_F32_EPR
#define GGML_F16_VEC        GGML_F32x4
#define GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F16_VEC_SET1   GGML_F32x4_SET1
#define GGML_F16_VEC_FMA    GGML_F32x4_FMA
#define GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#define GGML_F16_VEC_LOAD(p, i) (i & 0x1) ?                   \
  vec_extract_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_extract_fp32_from_shortl(vec_xl(0, p))
#define GGML_ENDIAN_BYTE(i) ((unsigned char *)&(uint16_t){1})[i]
#define GGML_F16_VEC_STORE(p, r, i)                             \
  if (i & 0x1)                                                  \
    vec_xst(vec_pack_to_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

#elif defined(__wasm_simd128__)

#define GGML_SIMD

// F32 WASM

#define GGML_F32_STEP 16
#define GGML_F32_EPR  4

#define GGML_F32x4              v128_t
#define GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#define GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#define GGML_F32x4_LOAD         wasm_v128_load
#define GGML_F32x4_STORE        wasm_v128_store
#define GGML_F32x4_FMA(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(b, c), a)
#define GGML_F32x4_ADD          wasm_f32x4_add
#define GGML_F32x4_MUL          wasm_f32x4_mul
#define GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    for (int i = 0; i < GGML_F32_ARR/2; ++i) {     \
        x[2*i] = wasm_f32x4_add(x[2*i], x[2*i+1]); \
    }                                              \
    for (int i = 0; i < GGML_F32_ARR/4; ++i) {     \
        x[4*i] = wasm_f32x4_add(x[4*i], x[4*i+2]); \
    }                                              \
    for (int i = 0; i < GGML_F32_ARR/8; ++i) {     \
        x[8*i] = wasm_f32x4_add(x[8*i], x[8*i+4]); \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#define GGML_F16_STEP 16
#define GGML_F16_EPR  4

inline static v128_t __wasm_f16x4_load(const ggml_fp16_t * p) {
    float tmp[4];

    tmp[0] = GGML_FP16_TO_FP32(p[0]);
    tmp[1] = GGML_FP16_TO_FP32(p[1]);
    tmp[2] = GGML_FP16_TO_FP32(p[2]);
    tmp[3] = GGML_FP16_TO_FP32(p[3]);

    return wasm_v128_load(tmp);
}

inline static void __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    float tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_FP32_TO_FP16(tmp[0]);
    p[1] = GGML_FP32_TO_FP16(tmp[1]);
    p[2] = GGML_FP32_TO_FP16(tmp[2]);
    p[3] = GGML_FP32_TO_FP16(tmp[3]);
}

#define GGML_F16x4             v128_t
#define GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#define GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#define GGML_F16x4_LOAD(x)     __wasm_f16x4_load(x)
#define GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#define GGML_F16x4_FMA         GGML_F32x4_FMA
#define GGML_F16x4_ADD         wasm_f32x4_add
#define GGML_F16x4_MUL         wasm_f32x4_mul
#define GGML_F16x4_REDUCE(res, x)                  \
{                                                  \
    for (int i = 0; i < GGML_F16_ARR/2; ++i) {     \
        x[2*i] = wasm_f32x4_add(x[2*i], x[2*i+1]); \
    }                                              \
    for (int i = 0; i < GGML_F16_ARR/4; ++i) {     \
        x[4*i] = wasm_f32x4_add(x[4*i], x[4*i+2]); \
    }                                              \
    for (int i = 0; i < GGML_F16_ARR/8; ++i) {     \
        x[8*i] = wasm_f32x4_add(x[8*i], x[8*i+4]); \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define GGML_F16_VEC                GGML_F16x4
#define GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#define GGML_F16_VEC_SET1           GGML_F16x4_SET1
#define GGML_F16_VEC_LOAD(p, i)     GGML_F16x4_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F16x4_FMA
#define GGML_F16_VEC_ADD            GGML_F16x4_ADD
#define GGML_F16_VEC_MUL            GGML_F16x4_MUL
#define GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#elif defined(__SSE3__)

#define GGML_SIMD

// F32 SSE

#define GGML_F32_STEP 32
#define GGML_F32_EPR  4

#define GGML_F32x4         __m128
#define GGML_F32x4_ZERO    _mm_setzero_ps()
#define GGML_F32x4_SET1(x) _mm_set1_ps(x)
#define GGML_F32x4_LOAD    _mm_loadu_ps
#define GGML_F32x4_STORE   _mm_storeu_ps
#if defined(__FMA__)
    // TODO: Does this work?
    #define GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#else
    #define GGML_F32x4_FMA(a, b, c) _mm_add_ps(_mm_mul_ps(b, c), a)
#endif
#define GGML_F32x4_ADD     _mm_add_ps
#define GGML_F32x4_MUL     _mm_mul_ps
#define GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    for (int i = 0; i < GGML_F32_ARR/2; ++i) {                    \
        x[2*i] = _mm_add_ps(x[2*i], x[2*i+1]);                    \
    }                                                             \
    for (int i = 0; i < GGML_F32_ARR/4; ++i) {                    \
        x[4*i] = _mm_add_ps(x[4*i], x[4*i+2]);                    \
    }                                                             \
    for (int i = 0; i < GGML_F32_ARR/8; ++i) {                    \
        x[8*i] = _mm_add_ps(x[8*i], x[8*i+4]);                    \
    }                                                             \
    const __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = _mm_cvtss_f32(_mm_hadd_ps(t0, t0));                     \
}
// TODO: is this optimal ?

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#define GGML_F16_STEP 32
#define GGML_F16_EPR  4

static inline __m128 __sse_f16x4_load(ggml_fp16_t *x) {
    float tmp[4];

    tmp[0] = GGML_FP16_TO_FP32(x[0]);
    tmp[1] = GGML_FP16_TO_FP32(x[1]);
    tmp[2] = GGML_FP16_TO_FP32(x[2]);
    tmp[3] = GGML_FP16_TO_FP32(x[3]);

    return _mm_loadu_ps(tmp);
}

static inline void __sse_f16x4_store(ggml_fp16_t *x, __m128 y) {
    float arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_FP32_TO_FP16(arr[0]);
    x[1] = GGML_FP32_TO_FP16(arr[1]);
    x[2] = GGML_FP32_TO_FP16(arr[2]);
    x[3] = GGML_FP32_TO_FP16(arr[3]);
}

#define GGML_F32Cx4             __m128
#define GGML_F32Cx4_ZERO        _mm_setzero_ps()
#define GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#define GGML_F32Cx4_LOAD(x)     __sse_f16x4_load(x)
#define GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#define GGML_F32Cx4_FMA         GGML_F32x4_FMA
#define GGML_F32Cx4_ADD         _mm_add_ps
#define GGML_F32Cx4_MUL         _mm_mul_ps
#define GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#define GGML_F16_VEC                 GGML_F32Cx4
#define GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#define GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#define GGML_F16_VEC_LOAD(p, i)      GGML_F32Cx4_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#define GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#define GGML_F16_VEC_ADD             GGML_F32Cx4_ADD
#define GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#define GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#endif

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#ifdef GGML_SIMD
#define GGML_F32_ARR (GGML_F32_STEP/GGML_F32_EPR)
#define GGML_F16_ARR (GGML_F16_STEP/GGML_F16_EPR)
#endif

//
// fundamental operations
//

inline static void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_set_i32(const int n, int32_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_set_f16(const int n, ggml_fp16_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void ggml_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void ggml_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }

inline static void ggml_vec_dot_f32(const int n, float * restrict s, const float * restrict x, const float * restrict y) {
#ifdef GGML_SIMD
    float sumf = 0.0f;
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

            sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}

inline static void ggml_vec_dot_f16(const int n, float * restrict s, ggml_fp16_t * restrict x, ggml_fp16_t * restrict y) {
    ggml_float sumf = 0.0;

#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

            sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
    }
#else
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
    }
#endif

    *s = sumf;
}

static void ggml_vec_dot_q4_0_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_0;

    assert(n % QK8_0 == 0);
    assert(nb % 2 == 0);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    float sumf = 0.0;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const block_q4_0 * restrict x0 = &x[i + 0];
        const block_q4_0 * restrict x1 = &x[i + 1];
        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b   = vdupq_n_u8(0xf);
        const int8x16_t  s8b   = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // interleave
        const int8x16_t v1_0ls = vuzp1q_s8(v1_0l, v1_0h);
        const int8x16_t v1_0hs = vuzp2q_s8(v1_0l, v1_0h);
        const int8x16_t v1_1ls = vuzp1q_s8(v1_1l, v1_1h);
        const int8x16_t v1_1hs = vuzp2q_s8(v1_1l, v1_1h);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls), v0_0hs, v1_0hs);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1ls), v0_1hs, v1_1hs);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), x1->d*y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0ls));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0ls));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0hs));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0hs));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1ls));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1ls));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1hs));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1hs));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), x1->d*y1->d);
#endif
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );

        __m256i bx = bytes_from_nibbles_32(x[i].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        bx = _mm256_sub_epi8( bx, off );

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        // Get absolute values of x vectors
        const __m256i ax = _mm256_sign_epi8(bx, bx);

        // Sign the values of the y vectors
        const __m256i sy = _mm256_sign_epi8(by, bx);

        // Perform multiplication and create 16-bit values
        const __m256i dot = _mm256_maddubs_epi16(ax, sy);

        const __m256i ones = _mm256_set1_epi16(1);
        __m256i xy_q = _mm256_madd_epi16(ones, dot);

        /* Convert to vectore of 8 int32_t to 8 floats */
        __m256 q = _mm256_cvtepi32_ps( xy_q );

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    // Return horizontal sum of the acc vector
    __m128 res = _mm256_extractf128_ps( acc, 1 );
    res = _mm_add_ps( res, _mm256_castps256_ps128( acc ) );
    res = _mm_add_ps( res, _mm_movehl_ps( res, res ) );
    res = _mm_add_ss( res, _mm_movehdup_ps( res ) );

    sumf = _mm_cvtss_f32( res );
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        const __m256 d = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );

        __m128i i32[2];
        for (int j = 0; j < 2; ++j) {
            // Load 8 bytes, and unpack 4 bit fields into bytes, making 16 bytes
            __m128i bx = bytes_from_nibbles_16(x[i].qs + 8*j);
            __m128i by = _mm_loadu_si128((const __m128i *)(y[i].qs + 16*j));

            // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
            const __m128i off = _mm_set1_epi8( 8 );
            bx = _mm_sub_epi8( bx, off );

            // Get absolute values of x vectors
            const __m128i ax = _mm_sign_epi8(bx, bx);

            // Sign the values of the y vectors
            const __m128i sy = _mm_sign_epi8(by, bx);

            // Perform multiplication and create 16-bit values
            const __m128i dot = _mm_maddubs_epi16(ax, sy);

            const __m128i ones = _mm_set1_epi16(1);
            i32[j] = _mm_madd_epi16(ones, dot);
        }

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps( _mm256_set_m128i( i32[0], i32[1] ));
        // Apply the scale, and accumulate
        acc = _mm256_add_ps(_mm256_mul_ps( d, p ), acc);
    }

    // Return horizontal sum of the acc vector
    __m128 res = _mm256_extractf128_ps( acc, 1 );
    res = _mm_add_ps( res, _mm256_castps256_ps128( acc ) );
    res = _mm_add_ps( res, _mm_movehl_ps( res, res ) );
    res = _mm_add_ss( res, _mm_movehdup_ps( res ) );

    sumf = _mm_cvtss_f32( res );
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        const uint8_t * restrict p0 = x[i].qs;
        const  int8_t * restrict p1 = y[i].qs;

        int sumi = 0;
        for (int j = 0; j < QK8_0/2; j++) {
            const uint8_t v0 = p0[j];

            const int i0 = (int8_t) (v0 & 0xf) - 8;
            const int i1 = (int8_t) (v0 >> 4)  - 8;

            const int i2 = p1[2*j + 0];
            const int i3 = p1[2*j + 1];

            sumi += i0*i2 + i1*i3;
        }
        sumf += d0*d1*sumi;
    }
#endif

    *s = sumf;
}

static void ggml_vec_dot_q4_1_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_0;

    assert(n % QK8_0 == 0);
    assert(nb % 2 == 0);

    const block_q4_1 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    float sumf = 0.0;

    // TODO: add AVX / WASM SIMD / etc
#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const block_q4_1 * restrict x0 = &x[i + 0];
        const block_q4_1 * restrict x1 = &x[i + 1];
        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0xf);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // interleave
        const int8x16_t v1_0ls = vuzp1q_s8(v1_0l, v1_0h);
        const int8x16_t v1_0hs = vuzp2q_s8(v1_0l, v1_0h);
        const int8x16_t v1_1ls = vuzp1q_s8(v1_1l, v1_1h);
        const int8x16_t v1_1hs = vuzp2q_s8(v1_1l, v1_1h);

        const int16x8_t s0i = vaddq_s16(
                        vaddq_s16(vmovl_s8(vget_low_s8(v1_0ls)), vmovl_s8(vget_high_s8(v1_0ls))),
                        vaddq_s16(vmovl_s8(vget_low_s8(v1_0hs)), vmovl_s8(vget_high_s8(v1_0hs))));

        const int16x8_t s1i = vaddq_s16(
                        vaddq_s16(vmovl_s8(vget_low_s8(v1_1ls)), vmovl_s8(vget_high_s8(v1_1ls))),
                        vaddq_s16(vmovl_s8(vget_low_s8(v1_1hs)), vmovl_s8(vget_high_s8(v1_1hs))));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddl_s16(vget_low_s16(s0i), vget_high_s16(s0i))), x0->m*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddl_s16(vget_low_s16(s1i), vget_high_s16(s1i))), x1->m*y1->d);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0l, v1_0ls), v0_0h, v1_0hs);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1l, v1_1ls), v0_1h, v1_1hs);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), x1->d*y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0l), vget_low_s8 (v1_0ls));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0l), vget_high_s8(v1_0ls));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0h), vget_low_s8 (v1_0hs));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0h), vget_high_s8(v1_0hs));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1l), vget_low_s8 (v1_1ls));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1l), vget_high_s8(v1_1ls));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1h), vget_low_s8 (v1_1hs));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1h), vget_high_s8(v1_1hs));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), x1->d*y1->d);
#endif
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        const float * d0 = &x[i].d;
        const float * d1 = &y[i].d;
        const float * m0 = &x[i].m;

        const __m256 d0v = _mm256_broadcast_ss( d0 );
        const __m256 d1v = _mm256_broadcast_ss( d1 );
        const __m256 m0v = _mm256_broadcast_ss( m0 );

        // Compute combined scales
        const __m256 d0d1 = _mm256_mul_ps( d0v, d1v );
        const __m256 d1m0 = _mm256_mul_ps( d1v, m0v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const __m256i bx = bytes_from_nibbles_32(x[i].qs);
        const __m256i by = _mm256_loadu_si256( (const __m256i *)y[i].qs );

        // Get absolute values of x vectors
        const __m256i ax = _mm256_sign_epi8( bx, bx );

        // Sign the values of the y vectors
        const __m256i sy = _mm256_sign_epi8( by, bx );

        // Perform multiplication and create 16-bit values
        const __m256i dot = _mm256_maddubs_epi16( ax, sy );
        const __m256i ones = _mm256_set1_epi16( 1 );
        const __m256i xy_q = _mm256_madd_epi16( ones, dot );

        // Convert to vector of 8 int32_t to 8 floats
        const __m256 xy = _mm256_cvtepi32_ps( xy_q );

        // Accumulate d0*d1*x*y
        acc = _mm256_fmadd_ps( d0d1, xy, acc );

        // Compute sum of y values
        const __m256i y16_l = _mm256_cvtepi8_epi16( _mm256_castsi256_si128( by ) );
        const __m256i y16_h = _mm256_cvtepi8_epi16( _mm256_extracti128_si256( by, 1 ) );
        const __m256i ysumi = _mm256_madd_epi16( _mm256_add_epi16(y16_l, y16_h), ones );
        const __m256 ysum = _mm256_cvtepi32_ps( ysumi );

        // Accumulate d1*m0*y
        acc = _mm256_fmadd_ps( d1m0, ysum, acc );
    }

    // Return horizontal sum of the acc vector
    __m128 res = _mm256_extractf128_ps( acc, 1 );
    res = _mm_add_ps( res, _mm256_castps256_ps128( acc ) );
    res = _mm_add_ps( res, _mm_movehl_ps( res, res ) );
    res = _mm_add_ss( res, _mm_movehdup_ps( res ) );

    sumf = _mm_cvtss_f32( res );
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float m0 = x[i].m;
        const float d1 = y[i].d;

        const uint8_t * restrict p0 = x[i].qs;
        const  int8_t * restrict p1 = y[i].qs;

        // TODO: this is very slow ..
        for (int j = 0; j < QK8_0/2; j++) {
            const uint8_t v0 = p0[j];

            const float f0 = d0*(v0 & 0xf) + m0;
            const float f1 = d0*(v0 >> 4)  + m0;

            const float f2 = d1*p1[2*j + 0];
            const float f3 = d1*p1[2*j + 1];

            sumf += f0*f2 + f1*f3;
        }
    }
#endif

    *s = sumf;
}

static void ggml_vec_dot_q4_2_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_0;

    assert(n % QK8_0 == 0);
    assert(nb % 2 == 0);
    assert(QK8_0 == 2*QK4_2);

    const block_q4_2 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    float sumf = 0.0;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const block_q4_2 * restrict x0_0 = &x[2*(i + 0) + 0];
        const block_q4_2 * restrict x0_1 = &x[2*(i + 0) + 1];
        const block_q4_2 * restrict x1_0 = &x[2*(i + 1) + 0];
        const block_q4_2 * restrict x1_1 = &x[2*(i + 1) + 1];

        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b   = vdupq_n_u8(0xf);
        const int8x16_t  s8b   = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vcombine_u8(vld1_u8(x0_0->qs), vld1_u8(x0_1->qs));
        const uint8x16_t v0_1 = vcombine_u8(vld1_u8(x1_0->qs), vld1_u8(x1_1->qs));

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // interleave
        const int8x16_t v0_0lz = vzip1q_s8(v0_0ls, v0_0hs);
        const int8x16_t v0_0hz = vzip2q_s8(v0_0ls, v0_0hs);
        const int8x16_t v0_1lz = vzip1q_s8(v0_1ls, v0_1hs);
        const int8x16_t v0_1hz = vzip2q_s8(v0_1ls, v0_1hs);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv0 = vmlaq_n_f32(sumv0, vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_0lz, v1_0l)), GGML_FP16_TO_FP32(x0_0->d)),
                vmulq_n_f32(vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_0hz, v1_0h)), GGML_FP16_TO_FP32(x0_1->d))), y0->d);

        sumv1 = vmlaq_n_f32(sumv1, vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_1lz, v1_1l)), GGML_FP16_TO_FP32(x1_0->d)),
                vmulq_n_f32(vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_1hz, v1_1h)), GGML_FP16_TO_FP32(x1_1->d))), y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0lz), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lz), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hz), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hz), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1lz), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lz), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hz), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hz), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(pl0), GGML_FP16_TO_FP32(x0_0->d)),
                vmulq_n_f32(vcvtq_f32_s32(ph0), GGML_FP16_TO_FP32(x0_1->d))), y0->d);

        sumv1 = vmlaq_n_f32(sumv1, vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(pl1), GGML_FP16_TO_FP32(x1_0->d)),
                vmulq_n_f32(vcvtq_f32_s32(ph1), GGML_FP16_TO_FP32(x1_1->d))), y1->d);
#endif
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; i++) {
        /* Compute combined scale for the block */
        const __m128 d0 = _mm_set1_ps(GGML_FP16_TO_FP32(x[2*i + 0].d));
        const __m128 d1 = _mm_set1_ps(GGML_FP16_TO_FP32(x[2*i + 1].d));
        const __m256 d = _mm256_mul_ps(_mm256_set_m128(d1, d0), _mm256_broadcast_ss(&y[i].d));

        __m128i bx0 = bytes_from_nibbles_16(x[2*i + 0].qs);
        __m128i bx1 = bytes_from_nibbles_16(x[2*i + 1].qs);
        __m256i bx = _mm256_set_m128i(bx1, bx0);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8(8);
        bx = _mm256_sub_epi8(bx, off);

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        // Get absolute values of x vectors
        const __m256i ax = _mm256_sign_epi8(bx, bx);
        // Sign the values of the y vectors
        const __m256i sy = _mm256_sign_epi8(by, bx);
        // Perform multiplication and create 16-bit values
        const __m256i dot = _mm256_maddubs_epi16(ax, sy);

        const __m256i ones = _mm256_set1_epi16(1);
        __m256i xy_q = _mm256_madd_epi16(ones, dot);

        /* Convert to vectore of 8 int32_t to 8 floats */
        __m256 q = _mm256_cvtepi32_ps(xy_q);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    // Return horizontal sum of the acc vector
    __m128 res = _mm256_extractf128_ps(acc, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(acc));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));

    sumf = _mm_cvtss_f32(res);
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const uint8_t * restrict x0 = x[2*i + 0].qs;
        const uint8_t * restrict x1 = x[2*i + 1].qs;
        const  int8_t * restrict y0 = y[i].qs;

        const float d0 = GGML_FP16_TO_FP32(x[2*i + 0].d);
        const float d1 = GGML_FP16_TO_FP32(x[2*i + 1].d);

        int sumi_0 = 0;
        int sumi_1 = 0;

        for (int j = 0; j < QK8_0/4; j++) {
            const uint8_t v0 = x0[j];
            const uint8_t v1 = x1[j];

            const int i0_0 = (int8_t) (v0 & 0xf) - 8;
            const int i1_0 = (int8_t) (v0 >> 4)  - 8;

            const int i0_1 = (int8_t) (v1 & 0xf) - 8;
            const int i1_1 = (int8_t) (v1 >> 4)  - 8;

            const int i2_0 = y0[2*j + 0];
            const int i3_0 = y0[2*j + 1];

            const int i2_1 = y0[2*(j + QK8_0/4) + 0];
            const int i3_1 = y0[2*(j + QK8_0/4) + 1];

            sumi_0 += i0_0*i2_0 + i1_0*i3_0;
            sumi_1 += i0_1*i2_1 + i1_1*i3_1;
        }

        sumf += (d0 * y[i].d) * sumi_0;
        sumf += (d1 * y[i].d) * sumi_1;
    }
#endif

    *s = sumf;
}

static void ggml_vec_dot_q4_3_q8_0(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_0;

    assert(n % QK8_0 == 0);
    assert(nb % 2 == 0);
    assert(QK8_0 == 2*QK4_2);

    const block_q4_3 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

    float sumf = 0.0;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const block_q4_3 * restrict x0_0 = &x[2*(i + 0) + 0];
        const block_q4_3 * restrict x0_1 = &x[2*(i + 0) + 1];
        const block_q4_3 * restrict x1_0 = &x[2*(i + 1) + 0];
        const block_q4_3 * restrict x1_1 = &x[2*(i + 1) + 1];

        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0xf);

        const float x0_0d = GGML_FP16_TO_FP32(x0_0->d);
        const float x0_1d = GGML_FP16_TO_FP32(x0_1->d);
        const float x1_0d = GGML_FP16_TO_FP32(x1_0->d);
        const float x1_1d = GGML_FP16_TO_FP32(x1_1->d);

        const float x0_0m = GGML_FP16_TO_FP32(x0_0->m);
        const float x0_1m = GGML_FP16_TO_FP32(x0_1->m);
        const float x1_0m = GGML_FP16_TO_FP32(x1_0->m);
        const float x1_1m = GGML_FP16_TO_FP32(x1_1->m);

        const uint8x16_t v0_0 = vcombine_u8(vld1_u8(x0_0->qs), vld1_u8(x0_1->qs));
        const uint8x16_t v0_1 = vcombine_u8(vld1_u8(x1_0->qs), vld1_u8(x1_1->qs));

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // interleave
        const int8x16_t v0_0lz = vzip1q_s8(v0_0l, v0_0h);
        const int8x16_t v0_0hz = vzip2q_s8(v0_0l, v0_0h);
        const int8x16_t v0_1lz = vzip1q_s8(v0_1l, v0_1h);
        const int8x16_t v0_1hz = vzip2q_s8(v0_1l, v0_1h);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        const int16x8_t sy0_0 = vaddq_s16(vmovl_s8(vget_low_s8(v1_0l)), vmovl_s8(vget_high_s8(v1_0l)));
        const int16x8_t sy0_1 = vaddq_s16(vmovl_s8(vget_low_s8(v1_0h)), vmovl_s8(vget_high_s8(v1_0h)));

        const int16x8_t sy1_0 = vaddq_s16(vmovl_s8(vget_low_s8(v1_1l)), vmovl_s8(vget_high_s8(v1_1l)));
        const int16x8_t sy1_1 = vaddq_s16(vmovl_s8(vget_low_s8(v1_1h)), vmovl_s8(vget_high_s8(v1_1h)));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddl_s16(vget_low_s16(sy0_0), vget_high_s16(sy0_0))), x0_0m*y0->d);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddl_s16(vget_low_s16(sy0_1), vget_high_s16(sy0_1))), x0_1m*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddl_s16(vget_low_s16(sy1_0), vget_high_s16(sy1_0))), x1_0m*y1->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddl_s16(vget_low_s16(sy1_1), vget_high_s16(sy1_1))), x1_1m*y1->d);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_0lz, v1_0l)), x0_0d*y0->d);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_0hz, v1_0h)), x0_1d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_1lz, v1_1l)), x1_0d*y1->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_1hz, v1_1h)), x1_1d*y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0lz), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lz), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hz), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hz), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1lz), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lz), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hz), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hz), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(pl0), x0_0d*y0->d);
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(ph0), x0_1d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(pl1), x1_0d*y1->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(ph1), x1_1d*y1->d);
#endif
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const uint8_t * restrict x0 = x[2*i + 0].qs;
        const uint8_t * restrict x1 = x[2*i + 1].qs;
        const  int8_t * restrict y0 = y[i].qs;

        const float d0 = GGML_FP16_TO_FP32(x[2*i + 0].d);
        const float m0 = GGML_FP16_TO_FP32(x[2*i + 0].m);
        const float d1 = GGML_FP16_TO_FP32(x[2*i + 1].d);
        const float m1 = GGML_FP16_TO_FP32(x[2*i + 1].m);

        int sy_0 = 0;
        int sy_1 = 0;

        int sxy_0 = 0;
        int sxy_1 = 0;

        for (int j = 0; j < QK8_0/4; j++) {
            const uint8_t v0 = x0[j];
            const uint8_t v1 = x1[j];

            const int x0_0 = v0 & 0xf;
            const int x1_0 = v0 >> 4;

            const int x0_1 = v1 & 0xf;
            const int x1_1 = v1 >> 4;

            const int y0_0 = y0[2*j + 0];
            const int y1_0 = y0[2*j + 1];

            const int y0_1 = y0[2*(j + QK8_0/4) + 0];
            const int y1_1 = y0[2*(j + QK8_0/4) + 1];

            sy_0 += y0_0 + y1_0;
            sy_1 += y0_1 + y1_1;

            sxy_0 += x0_0*y0_0 + x1_0*y1_0;
            sxy_1 += x0_1*y0_1 + x1_1*y1_1;
        }

        sumf += (d0*sxy_0 + m0*sy_0)*y[i].d;
        sumf += (d1*sxy_1 + m1*sy_1)*y[i].d;
    }
#endif

    *s = sumf;
}


// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void ggml_vec_dot_f16_unroll(const int n, const int xs, float * restrict s, void * restrict xv, ggml_fp16_t * restrict y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * restrict x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC sum[GGML_VEC_DOT_UNROLL][GGML_F16_ARR] = { { GGML_F16_VEC_ZERO } };

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

            for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
                ax[j] = GGML_F16_VEC_LOAD(x[k] + i + j*GGML_F16_EPR, j);

                sum[k][j] = GGML_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
            }
        }
    }

    // reduce sum0..sum3 to sum0
    for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
        GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float)(GGML_FP16_TO_FP32(x[j][i])*GGML_FP16_TO_FP32(y[i]));
        }
    }
#else
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float)(GGML_FP16_TO_FP32(x[j][i])*GGML_FP16_TO_FP32(y[i]));
        }
    }
#endif

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = sumf[i];
    }
}

inline static void ggml_vec_mad_f32(const int n, float * restrict y, const float * restrict x, const float v) {
#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

            GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] += x[i]*v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

//inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

            GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void ggml_vec_norm_f32 (const int n, float * s, const float * x) { ggml_vec_dot_f32(n, s, x, x); *s = sqrtf(*s);   }
inline static void ggml_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
inline static void ggml_vec_abs_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void ggml_vec_sgn_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void ggml_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void ggml_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }

static const float GELU_COEF_A    = 0.044715f;
static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

inline static float ggml_gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

inline static void ggml_vec_gelu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = table_gelu_f16[i16[i]];
    }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = GGML_FP16_TO_FP32(table_gelu_f16[t]);
    }
}
#else
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]);
    }
}
#endif

// Sigmoid Linear Unit (SiLU) function
inline static float ggml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}

inline static void ggml_vec_silu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = table_silu_f16[i16[i]];
    }
}

#ifdef GGML_SILU_FP16
inline static void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = GGML_FP16_TO_FP32(table_silu_f16[t]);
    }
}
#else
inline static void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}
#endif

inline static void ggml_vec_sum_f32(const int n, float * s, const float * x) {
#ifndef GGML_USE_ACCELERATE
    ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = sum;
#else
    vDSP_sve(x, 1, s, n);
#endif
}

inline static void ggml_vec_max_f32(const int n, float * s, const float * x) {
#ifndef GGML_USE_ACCELERATE
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}

inline static void ggml_vec_norm_inv_f32(const int n, float * s, const float * x) {
    ggml_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

//
// logging
//

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

//
// data types
//

static const int GGML_BLCK_SIZE[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]  = 1,
    [GGML_TYPE_F16]  = 1,
    [GGML_TYPE_Q4_0] = QK4_0,
    [GGML_TYPE_Q4_1] = QK4_1,
    [GGML_TYPE_Q4_2] = QK4_2,
    [GGML_TYPE_Q4_3] = QK4_3,
    [GGML_TYPE_Q8_0] = QK8_0,
    [GGML_TYPE_I8]   = 1,
    [GGML_TYPE_I16]  = 1,
    [GGML_TYPE_I32]  = 1,
};
static_assert(GGML_TYPE_COUNT == 10, "GGML_BLCK_SIZE is outdated");

static const size_t GGML_TYPE_SIZE[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]  = sizeof(float),
    [GGML_TYPE_F16]  = sizeof(ggml_fp16_t),
    [GGML_TYPE_Q4_0] = sizeof(block_q4_0),
    [GGML_TYPE_Q4_1] = sizeof(block_q4_1),
    [GGML_TYPE_Q4_2] = sizeof(block_q4_2),
    [GGML_TYPE_Q4_3] = sizeof(block_q4_3),
    [GGML_TYPE_Q8_0] = sizeof(block_q8_0),
    [GGML_TYPE_I8]   = sizeof(int8_t),
    [GGML_TYPE_I16]  = sizeof(int16_t),
    [GGML_TYPE_I32]  = sizeof(int32_t),
};
static_assert(GGML_TYPE_COUNT == 10, "GGML_TYPE_SIZE is outdated");


static const char * GGML_TYPE_NAME[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]  = "f32",
    [GGML_TYPE_F16]  = "f16",
    [GGML_TYPE_Q4_0] = "q4_0",
    [GGML_TYPE_Q4_1] = "q4_1",
    [GGML_TYPE_Q4_2] = "q4_2",
    [GGML_TYPE_Q4_3] = "q4_3",
    [GGML_TYPE_Q8_0] = "q8_0",
    [GGML_TYPE_I8]   = "i8",
    [GGML_TYPE_I16]  = "i16",
    [GGML_TYPE_I32]  = "i32",
};
static_assert(GGML_TYPE_COUNT == 10, "GGML_TYPE_NAME is outdated");

static bool GGML_IS_QUANTIZED[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32]  = false,
    [GGML_TYPE_F16]  = false,
    [GGML_TYPE_Q4_0] = true,
    [GGML_TYPE_Q4_1] = true,
    [GGML_TYPE_Q4_2] = true,
    [GGML_TYPE_Q4_3] = true,
    [GGML_TYPE_Q8_0] = true,
    [GGML_TYPE_I8]   = false,
    [GGML_TYPE_I16]  = false,
    [GGML_TYPE_I32]  = false,
};
static_assert(GGML_TYPE_COUNT == 10, "GGML_IS_QUANTIZED is outdated");

static const char * GGML_OP_LABEL[GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "SUM",
    "MEAN",
    "REPEAT",
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "RELU",
    "GELU",
    "SILU",
    "NORM",
    "RMS_NORM",

    "MUL_MAT",

    "SCALE",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "DIAG_MASK_INF",
    "SOFT_MAX",
    "ROPE",
    "CONV_1D_1S",
    "CONV_1D_2S",

    "FLASH_ATTN",
    "FLASH_FF",

    "MAP_UNARY",
    "MAP_BINARY",
};

static_assert(GGML_OP_COUNT == 38, "GGML_OP_COUNT != 38");

static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "Σx",
    "Σx/n",
    "repeat(x)",
    "abs(x)",
    "sgn(x)",
    "-x",
    "step(x)",
    "relu(x)",
    "gelu(x)",
    "silu(x)",
    "norm(x)",
    "rms_norm(x)",

    "X*Y",

    "x*v",
    "x-\\>y",
    "cont(x)",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "diag_mask_inf(x)",
    "soft_max(x)",
    "rope(x)",
    "conv_1d_1s(x)",
    "conv_1d_2s(x)",

    "flash_attn(x)",
    "flash_ff(x)",

    "f(x)",
    "f(x,y)",
};

static_assert(GGML_OP_COUNT == 38, "GGML_OP_COUNT != 38");

static_assert(sizeof(struct ggml_object)%GGML_MEM_ALIGN == 0, "ggml_object size must be a multiple of GGML_MEM_ALIGN");
static_assert(sizeof(struct ggml_tensor)%GGML_MEM_ALIGN == 0, "ggml_tensor size must be a multiple of GGML_MEM_ALIGN");

//
// ggml context
//

struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;

    struct ggml_scratch scratch;
    struct ggml_scratch scratch_save;
};

struct ggml_context_container {
    bool used;

    struct ggml_context context;
};

//
// compute types
//

enum ggml_task_type {
    GGML_TASK_INIT = 0,
    GGML_TASK_COMPUTE,
    GGML_TASK_FINALIZE,
};

struct ggml_compute_params {
    enum ggml_task_type type;

    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;
};

//
// ggml state
//

struct ggml_state {
    struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
};

// global state
static struct ggml_state g_state;
static atomic_int g_state_barrier = 0;

// barrier via spin lock
inline static void ggml_critical_section_start(void) {
    int processing = atomic_fetch_add(&g_state_barrier, 1);

    while (processing > 0) {
        // wait for other threads to finish
        atomic_fetch_sub(&g_state_barrier, 1);
        sched_yield(); // TODO: reconsider this
        processing = atomic_fetch_add(&g_state_barrier, 1);
    }
}

// TODO: make this somehow automatically executed
//       some sort of "sentry" mechanism
inline static void ggml_critical_section_end(void) {
    atomic_fetch_sub(&g_state_barrier, 1);
}

////////////////////////////////////////////////////////////////////////////////

void ggml_print_object(const struct ggml_object * obj) {
    GGML_PRINT(" - ggml_object: offset = %zu, size = %zu, next = %p\n",
            obj->offs, obj->size, (const void *) obj->next);
}

void ggml_print_objects(const struct ggml_context * ctx) {
    struct ggml_object * obj = ctx->objects_begin;

    GGML_PRINT("%s: objects in context %p:\n", __func__, (const void *) ctx);

    while (obj != NULL) {
        ggml_print_object(obj);
        obj = obj->next;
    }

    GGML_PRINT("%s: --- end ---\n", __func__);
}

int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return (ggml_nelements(tensor)*GGML_TYPE_SIZE[tensor->type])/GGML_BLCK_SIZE[tensor->type];
}

int ggml_blck_size(enum ggml_type type) {
    return GGML_BLCK_SIZE[type];
}

size_t ggml_type_size(enum ggml_type type) {
    return GGML_TYPE_SIZE[type];
}

float ggml_type_sizef(enum ggml_type type) {
    return ((float)(GGML_TYPE_SIZE[type]))/GGML_BLCK_SIZE[type];
}

const char * ggml_type_name(enum ggml_type type) {
    return GGML_TYPE_NAME[type];
}


size_t ggml_element_size(const struct ggml_tensor * tensor) {
    return GGML_TYPE_SIZE[tensor->type];
}

static inline bool ggml_is_scalar(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ggml_is_vector(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ggml_is_matrix(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        (t0->ne[0] == t1->ne[0])  &&
        (t0->ne[2] == t1->ne[2])  &&
        (t0->ne[3] == t1->ne[3]);
}

bool ggml_is_quantized(enum ggml_type type) {
    return GGML_IS_QUANTIZED[type];
}

static inline bool ggml_is_transposed(const struct ggml_tensor * tensor) {
    return tensor->nb[0] > tensor->nb[1];
}

static inline bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == GGML_TYPE_SIZE[tensor->type] &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/GGML_BLCK_SIZE[tensor->type] &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static inline bool ggml_is_padded_1d(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == GGML_TYPE_SIZE[tensor->type] &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static inline bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        (t0->ne[0] == t1->ne[0] ) &&
        (t0->ne[1] == t1->ne[1] ) &&
        (t0->ne[2] == t1->ne[2] ) &&
        (t0->ne[3] == t1->ne[3] );
}

// check if t1 can be represented as a repeatition of t0
static inline bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

static inline int ggml_up32(int n) {
    return (n + 31) & ~31;
}

static inline int ggml_up64(int n) {
    return (n + 63) & ~63;
}

static inline int ggml_up(int n, int m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    return (n + m - 1) & ~(m - 1);
}

// assert that pointer is aligned to GGML_MEM_ALIGN
#define ggml_assert_aligned(ptr) \
    GGML_ASSERT(((uintptr_t) (ptr))%GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

struct ggml_context * ggml_init(struct ggml_init_params params) {
    // make this function thread safe
    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize time system (required on Windows)
        ggml_time_init();

        // initialize GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            ggml_fp16_t ii;
            for (int i = 0; i < (1 << 16); ++i) {
                uint16_t ui = i;
                memcpy(&ii, &ui, sizeof(ii));
                const float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
                table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
                table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
                table_exp_f16[i]  = GGML_FP32_TO_FP16(expf(f));
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        // initialize g_state
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            g_state = (struct ggml_state) {
                /*.contexts =*/ { { 0 } },
            };

            for (int i = 0; i < GGML_MAX_CONTEXTS; ++i) {
                g_state.contexts[i].used = false;
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        // initialize cuBLAS
        #if defined(GGML_USE_CUBLAS)
        init_cublas();
        #endif

        is_first_call = false;
    }

    // find non-used context in g_state
    struct ggml_context * ctx = NULL;

    for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
        if (!g_state.contexts[i].used) {
            g_state.contexts[i].used = true;
            ctx = &g_state.contexts[i].context;

            GGML_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
            break;
        }
    }

    if (ctx == NULL) {
        GGML_PRINT_DEBUG("%s: no unused context found\n", __func__);

        ggml_critical_section_end();

        return NULL;
    }

    const size_t mem_size = (params.mem_size + GGML_MEM_ALIGN - 1) & ~(GGML_MEM_ALIGN - 1);

    *ctx = (struct ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : GGML_ALIGNED_MALLOC(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
        /*.scratch            =*/ { 0, 0, NULL, },
        /*.scratch_save       =*/ { 0, 0, NULL, },
    };

    GGML_ASSERT(ctx->mem_buffer != NULL);

    ggml_assert_aligned(ctx->mem_buffer);

    GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    ggml_critical_section_end();

    return ctx;
}

void ggml_free(struct ggml_context * ctx) {
    // make this function thread safe
    ggml_critical_section_start();

    bool found = false;

    for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
        if (&g_state.contexts[i].context == ctx) {
            g_state.contexts[i].used = false;

            GGML_PRINT_DEBUG("%s: context %d with %d objects has been freed. memory used = %zu\n",
                    __func__, i, ctx->n_objects, ctx->objects_end->offs + ctx->objects_end->size);

            if (ctx->mem_buffer_owned) {
                GGML_ALIGNED_FREE(ctx->mem_buffer);
            }

            found = true;
            break;
        }
    }

    if (!found) {
        GGML_PRINT_DEBUG("%s: context not found\n", __func__);
    }

    ggml_critical_section_end();
}

size_t ggml_used_mem(const struct ggml_context * ctx) {
    return ctx->objects_end->offs + ctx->objects_end->size;
}

size_t ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch) {
    const size_t result = ctx->scratch.data ? ctx->scratch.offs : 0;

    ctx->scratch = scratch;

    return result;
}

////////////////////////////////////////////////////////////////////////////////

struct ggml_tensor * ggml_new_tensor_impl(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    n_dims,
        const int64_t* ne,
        void*  data) {
    // always insert objects at the end of the context's memory pool
    struct ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    size_t size_needed = 0;

    if (data == NULL && !ctx->no_alloc) {
        size_needed += GGML_TYPE_SIZE[type]*(ne[0]/GGML_BLCK_SIZE[type]);
        for (int i = 1; i < n_dims; i++) {
            size_needed *= ne[i];
        }
        // align to GGML_MEM_ALIGN
        size_needed = ((size_needed + GGML_MEM_ALIGN - 1)/GGML_MEM_ALIGN)*GGML_MEM_ALIGN;
    }

    char * const mem_buffer = ctx->mem_buffer;
    struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

    if (ctx->scratch.data == NULL || data != NULL) {
        size_needed += sizeof(struct ggml_tensor);

        if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
            GGML_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                    __func__, cur_end + size_needed + GGML_OBJECT_SIZE, ctx->mem_size);
            assert(false);
            return NULL;
        }

        *obj_new = (struct ggml_object) {
            .offs = cur_end + GGML_OBJECT_SIZE,
            .size = size_needed,
            .next = NULL,
        };
    } else {
        if (ctx->scratch.offs + size_needed > ctx->scratch.size) {
            GGML_PRINT("%s: not enough space in the scratch memory\n", __func__);
            assert(false);
            return NULL;
        }

        if (cur_end + sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE > ctx->mem_size) {
            GGML_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                    __func__, cur_end + sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE, ctx->mem_size);
            assert(false);
            return NULL;
        }

        data = (char * const) ctx->scratch.data + ctx->scratch.offs;

        *obj_new = (struct ggml_object) {
            .offs = cur_end + GGML_OBJECT_SIZE,
            .size = sizeof(struct ggml_tensor),
            .next = NULL,
        };

        //printf("scratch offs = %zu, size_needed = %zu\n", ctx->scratch.offs, size_needed);

        ctx->scratch.offs += size_needed;
    }

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    struct ggml_tensor * const result = (struct ggml_tensor *)(mem_buffer + obj_new->offs);

    ggml_assert_aligned(result);

    *result = (struct ggml_tensor) {
        /*.type         =*/ type,
        /*.n_dims       =*/ n_dims,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ GGML_OP_NONE,
        /*.is_param     =*/ false,
        /*.grad         =*/ NULL,
        /*.src0         =*/ NULL,
        /*.src1         =*/ NULL,
        /*.opt          =*/ { NULL },
        /*.n_tasks      =*/ 0,
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
        /*.data         =*/ (data == NULL && !ctx->no_alloc) ? (void *)(result + 1) : data,
        /*.pad          =*/ { 0 },
    };

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //ggml_assert_aligned(result->data);

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = GGML_TYPE_SIZE[type];
    result->nb[1] = result->nb[0]*(result->ne[0]/GGML_BLCK_SIZE[type]);
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    return result;
}

struct ggml_tensor * ggml_new_tensor(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    n_dims,
        const int64_t * ne) {
    return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL);
}

struct ggml_tensor * ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0) {
    return ggml_new_tensor(ctx, type, 1, &ne0);
}

struct ggml_tensor * ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1) {
    const int64_t ne[2] = { ne0, ne1 };
    return ggml_new_tensor(ctx, type, 2, ne);
}

struct ggml_tensor * ggml_new_tensor_3d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    return ggml_new_tensor(ctx, type, 3, ne);
}

struct ggml_tensor * ggml_new_tensor_4d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    return ggml_new_tensor(ctx, type, 4, ne);
}

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ctx->scratch = ctx->scratch_save;

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ctx->scratch = ctx->scratch_save;

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src) {
    return ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, NULL);
}

struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor) {
    memset(tensor->data, 0, ggml_nbytes(tensor));
    return tensor;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return 0.0f;
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return 0.0f;
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

void * ggml_get_data(const struct ggml_tensor * tensor) {
    return tensor->data;
}

float * ggml_get_data_f32(const struct ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);
    return (float *)(tensor->data);
}

struct ggml_tensor * ggml_view_tensor(
        struct ggml_context * ctx,
        const struct ggml_tensor * src) {
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data);

    result->nb[0] = src->nb[0];
    result->nb[1] = src->nb[1];
    result->nb[2] = src->nb[2];
    result->nb[3] = src->nb[3];

    return result;
}

////////////////////////////////////////////////////////////////////////////////

// ggml_dup

struct ggml_tensor * ggml_dup_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_DUP;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_dup(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    return ggml_dup_impl(ctx, a, false);
}

struct ggml_tensor * ggml_dup_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    return ggml_dup_impl(ctx, a, true);
}

// ggml_add

struct ggml_tensor * ggml_add_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_ADD;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_add_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_add_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_add_impl(ctx, a, b, true);
}

// ggml_sub

struct ggml_tensor * ggml_sub_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_SUB;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_sub(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_sub_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_sub_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

struct ggml_tensor * ggml_mul_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    if (inplace) {
        GGML_ASSERT(is_node == false);
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_MUL;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_mul(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_mul_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_mul_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

struct ggml_tensor * ggml_div_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    if (inplace) {
        GGML_ASSERT(is_node == false);
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_DIV;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_div(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_div_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_div_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

struct ggml_tensor * ggml_sqr_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_SQR;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_sqr(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sqr_impl(ctx, a, false);
}

struct ggml_tensor * ggml_sqr_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

struct ggml_tensor * ggml_sqrt_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_SQRT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_sqrt(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sqrt_impl(ctx, a, false);
}

struct ggml_tensor * ggml_sqrt_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sqrt_impl(ctx, a, true);
}

// ggml_sum

struct ggml_tensor * ggml_sum(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op   = GGML_OP_SUM;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_mean

struct ggml_tensor * ggml_mean(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        GGML_ASSERT(false); // TODO: implement
        is_node = true;
    }

    int64_t ne[GGML_MAX_DIMS] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, a->n_dims, ne);

    result->op   = GGML_OP_MEAN;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_repeat

struct ggml_tensor * ggml_repeat(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    GGML_ASSERT(ggml_can_repeat(a, b));

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    if (ggml_are_same_shape(a, b) && !is_node) {
        return a;
    }

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);

    result->op   = GGML_OP_REPEAT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_abs

struct ggml_tensor * ggml_abs_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_ABS;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_abs(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_abs_impl(ctx, a, false);
}

struct ggml_tensor * ggml_abs_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_abs_impl(ctx, a, true);
}


// ggml_sgn

struct ggml_tensor * ggml_sgn_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_SGN;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_sgn(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sgn_impl(ctx, a, false);
}

struct ggml_tensor * ggml_sgn_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sgn_impl(ctx, a, true);
}

// ggml_neg

struct ggml_tensor * ggml_neg_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_NEG;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_neg(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_neg_impl(ctx, a, false);
}

struct ggml_tensor * ggml_neg_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_neg_impl(ctx, a, true);
}

// ggml_step

struct ggml_tensor * ggml_step_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_STEP;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_step(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_step_impl(ctx, a, false);
}

struct ggml_tensor * ggml_step_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_step_impl(ctx, a, true);
}

// ggml_relu

struct ggml_tensor * ggml_relu_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_RELU;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_relu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_relu_impl(ctx, a, false);
}

struct ggml_tensor * ggml_relu_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_relu_impl(ctx, a, true);
}

// ggml_gelu

struct ggml_tensor * ggml_gelu_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_GELU;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_gelu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_gelu_impl(ctx, a, false);
}

struct ggml_tensor * ggml_gelu_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_gelu_impl(ctx, a, true);
}

// ggml_silu

struct ggml_tensor * ggml_silu_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_SILU;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_silu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_silu_impl(ctx, a, false);
}

struct ggml_tensor * ggml_silu_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_silu_impl(ctx, a, true);
}

// ggml_norm

struct ggml_tensor * ggml_norm_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_NORM;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store epsilon here?

    return result;
}

struct ggml_tensor * ggml_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_norm_impl(ctx, a, false);
}

struct ggml_tensor * ggml_norm_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_norm_impl(ctx, a, true);
}

struct ggml_tensor * ggml_rms_norm_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_RMS_NORM;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store epsilon here?

    return result;
}

struct ggml_tensor * ggml_rms_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_rms_norm_impl(ctx, a, false);
}

struct ggml_tensor * ggml_rms_norm_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_rms_norm_impl(ctx, a, true);
}

// ggml_mul_mat

struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    const int64_t ne[4] = { a->ne[1], b->ne[1], a->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, MIN(a->n_dims, b->n_dims), ne);

    result->op   = GGML_OP_MUL_MAT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_scale

struct ggml_tensor * ggml_scale_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool inplace) {
    GGML_ASSERT(ggml_is_scalar(b));
    GGML_ASSERT(ggml_is_padded_1d(a));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op   = GGML_OP_SCALE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_scale(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_scale_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_scale_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_scale_impl(ctx, a, b, true);
}

// ggml_cpy

struct ggml_tensor * ggml_cpy_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool inplace) {
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // make a view of the destination
    struct ggml_tensor * result = ggml_view_tensor(ctx, b);

    result->op   = GGML_OP_CPY;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_cpy(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_cpy_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_cpy_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_cpy_impl(ctx, a, b, true);
}

// ggml_cont

struct ggml_tensor * ggml_cont_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && a->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = GGML_OP_CONT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_cont(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    return ggml_cont_impl(ctx, a, false);
}

struct ggml_tensor * ggml_cont_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    return ggml_cont_impl(ctx, a, true);
}

// ggml_reshape

struct ggml_tensor * ggml_reshape(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(b));
    GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

    bool is_node = false;

    if (a->grad || b->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, b->n_dims, b->ne, a->data);

    result->op   = GGML_OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    bool is_node = false;

    if (a->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[2] = { ne0, ne1 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a->data);

    result->op   = GGML_OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_reshape_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    bool is_node = false;

    if (a->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[3] = { ne0, ne1, ne2 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a->data);

    result->op   = GGML_OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_view_1d

struct ggml_tensor * ggml_view_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    if (a->grad) {
        GGML_ASSERT(false); // gradient propagation is not supported
    }

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, &ne0, (char *) a->data + offset);

    result->op   = GGML_OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the offset here?

    return result;
}

// ggml_view_2d

struct ggml_tensor * ggml_view_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    if (a->grad) {
        GGML_ASSERT(false); // gradient propagation is not supported
    }

    const int64_t ne[GGML_MAX_DIMS] = { ne0, ne1, 1, 1 };

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, (char *) a->data + offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    result->op   = GGML_OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the offset here?

    return result;
}

// ggml_view_3d

struct ggml_tensor * ggml_view_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    if (a->grad) {
        GGML_ASSERT(false); // gradient propagation is not supported
    }

    const int64_t ne[GGML_MAX_DIMS] = { ne0, ne1, ne2, 1 };

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, (char *) a->data + offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    result->op   = GGML_OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the offset here?

    return result;
}

// ggml_permute

struct ggml_tensor * ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3) {
    GGML_ASSERT(axis0 >= 0 && axis0 < GGML_MAX_DIMS);
    GGML_ASSERT(axis1 >= 0 && axis1 < GGML_MAX_DIMS);
    GGML_ASSERT(axis2 >= 0 && axis2 < GGML_MAX_DIMS);
    GGML_ASSERT(axis3 >= 0 && axis3 < GGML_MAX_DIMS);

    GGML_ASSERT(axis0 != axis1);
    GGML_ASSERT(axis0 != axis2);
    GGML_ASSERT(axis0 != axis3);
    GGML_ASSERT(axis1 != axis2);
    GGML_ASSERT(axis1 != axis3);
    GGML_ASSERT(axis2 != axis3);

    bool is_node = false;

    if (a->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    int ne[GGML_MAX_DIMS];
    int nb[GGML_MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    result->ne[0] = ne[0];
    result->ne[1] = ne[1];
    result->ne[2] = ne[2];
    result->ne[3] = ne[3];

    result->nb[0] = nb[0];
    result->nb[1] = nb[1];
    result->nb[2] = nb[2];
    result->nb[3] = nb[3];

    result->op   = GGML_OP_PERMUTE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the permutation here?

    return result;
}

// ggml_transpose

struct ggml_tensor * ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op   = GGML_OP_TRANSPOSE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_get_rows

struct ggml_tensor * ggml_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);

    bool is_node = false;

    if (a->grad || b->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a->ne[0], b->ne[0]);

    result->op   = GGML_OP_GET_ROWS;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_diag_mask_inf

struct ggml_tensor * ggml_diag_mask_inf(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_past) {
    bool is_node = false;

    if (a->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);
    struct ggml_tensor * b = ggml_new_i32(ctx, n_past);

    result->op   = GGML_OP_DIAG_MASK_INF;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_soft_max

struct ggml_tensor * ggml_soft_max(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op   = GGML_OP_SOFT_MAX;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_rope

struct ggml_tensor * ggml_rope(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_past,
        int                   n_dims,
        int                   mode) {
    GGML_ASSERT(n_past >= 0);
    bool is_node = false;

    if (a->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ((int32_t *) b->data)[0] = n_past;
    ((int32_t *) b->data)[1] = n_dims;
    ((int32_t *) b->data)[2] = mode;

    result->op   = GGML_OP_ROPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_conv_1d_1s

struct ggml_tensor * ggml_conv_1d_1s(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[1] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);
    bool is_node = false;

    if (a->grad || b->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[4] = { b->ne[0], a->ne[2], 1, 1, };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);

    result->op   = GGML_OP_CONV_1D_1S;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_conv_1d_2s

struct ggml_tensor * ggml_conv_1d_2s(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_is_matrix(b));
    GGML_ASSERT(a->ne[1] == b->ne[1]);
    GGML_ASSERT(a->ne[3] == 1);
    bool is_node = false;

    if (a->grad || b->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[4] = { b->ne[0]/2, a->ne[2], 1, 1, };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);

    result->op   = GGML_OP_CONV_1D_2S;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_flash_attn

struct ggml_tensor * ggml_flash_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        bool                  masked) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    bool is_node = false;

    if (q->grad || k->grad || v->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    //struct ggml_tensor * result = ggml_dup_tensor(ctx, q);
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, q->ne);

    result->op   = GGML_OP_FLASH_ATTN;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = q;
    result->src1 = k;
    result->opt[0] = v;
    result->opt[1] = ggml_new_i32(ctx, masked ? 1 : 0);

    return result;
}

// ggml_flash_ff

struct ggml_tensor * ggml_flash_ff(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b0,
        struct ggml_tensor  * b1,
        struct ggml_tensor  * c0,
        struct ggml_tensor  * c1) {
    GGML_ASSERT(ggml_can_mul_mat(b0, a));
    // TODO: more checks

    bool is_node = false;

    if (a->grad || b0->grad || b1->grad || c0->grad || c1->grad) {
        GGML_ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    //struct ggml_tensor * result = ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, a->ne);

    result->op   = GGML_OP_FLASH_FF;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b0;
    result->opt[0] = b1;
    result->opt[1] = c0;
    result->opt[2] = c1;

    return result;
}

// ggml_map_unary

struct ggml_tensor * ggml_map_unary_impl_f32(
        struct ggml_context        * ctx,
        struct ggml_tensor         * a,
        const  ggml_unary_op_f32_t fun,
        bool   inplace) {
    bool is_node = false;

    if (!inplace && a->grad) {
        is_node = true;
    }

    struct ggml_tensor * addr_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sizeof(void *) / sizeof(int32_t));
    *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;
    struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op = GGML_OP_MAP_UNARY;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->opt[0] = addr_tensor;

    return result;
}

struct ggml_tensor * ggml_map_unary_f32(
        struct ggml_context        * ctx,
        struct ggml_tensor         * a,
        const  ggml_unary_op_f32_t fun) {
    return ggml_map_unary_impl_f32(ctx, a, fun, false);
}

struct ggml_tensor * ggml_map_unary_inplace_f32(
        struct ggml_context        * ctx,
        struct ggml_tensor         * a,
        const  ggml_unary_op_f32_t fun) {
    return ggml_map_unary_impl_f32(ctx, a, fun, true);
}

// ggml_map_binary

struct ggml_tensor * ggml_map_binary_impl_f32(
        struct ggml_context         * ctx,
        struct ggml_tensor          * a,
        struct ggml_tensor          * b,
        const  ggml_binary_op_f32_t fun,
        bool   inplace) {
    GGML_ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor * addr_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sizeof(void *) / sizeof(int32_t));
    *((void (**)(void))addr_tensor->data) = (void (*)(void))fun;
    struct ggml_tensor *result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op = GGML_OP_MAP_BINARY;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;
    result->opt[0] = addr_tensor;

    return result;
}

struct ggml_tensor * ggml_map_binary_f32(
        struct ggml_context         * ctx,
        struct ggml_tensor          * a,
        struct ggml_tensor          * b,
        const  ggml_binary_op_f32_t fun) {
    return ggml_map_binary_impl_f32(ctx, a, b, fun, false);
}

struct ggml_tensor * ggml_map_binary_inplace_f32(
        struct ggml_context         * ctx,
        struct ggml_tensor          * a,
        struct ggml_tensor          * b,
        const  ggml_binary_op_f32_t fun) {
    return ggml_map_binary_impl_f32(ctx, a, b, fun, true);
}

////////////////////////////////////////////////////////////////////////////////

void ggml_set_param(
        struct ggml_context * ctx,
        struct ggml_tensor * tensor) {
    tensor->is_param = true;

    GGML_ASSERT(tensor->grad == NULL);
    tensor->grad = ggml_dup_tensor(ctx, tensor);
}

// ggml_compute_forward_dup

static void ggml_compute_forward_dup_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
        // parallelize by elements
        const int ne = ggml_nelements(dst);
        const int dr = (ne + nth - 1) / nth;
        const int ie0 = dr * ith;
        const int ie1 = MIN(ie0 + dr, ne);

        memcpy(
            ((char *)  dst->data + ie0*nb0),
            ((char *) src0->data + ie0*nb00),
            (ie1 - ie0) * GGML_TYPE_SIZE[src0->type]);

        return;
    }

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == GGML_TYPE_SIZE[src0->type] && nb0 == GGML_TYPE_SIZE[dst->type]) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

    if (ggml_is_contiguous(dst)) {
        if (nb00 == sizeof(ggml_fp16_t)) {
            if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = GGML_FP16_TO_FP32(src0_ptr[i00]);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (ggml_is_quantized(dst->type)) {
                quantize_row_q_t const quantize_row_q = quantize_fns[dst->type].quantize_row_q;
                float * src0_f32 = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / GGML_BLCK_SIZE[dst->type]);
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                            for (int i00 = 0; i00 < ne00; i00++) {
                                src0_f32[i00] = GGML_FP16_TO_FP32(src0_ptr[i00]);
                            }

                            quantize_row_q(src0_f32, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                GGML_ASSERT(false); // TODO: implement
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = GGML_FP16_TO_FP32(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else {
                GGML_ASSERT(false); // TODO: implement
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(ggml_fp16_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(float *) dst_ptr = GGML_FP16_TO_FP32(*(const ggml_fp16_t *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        GGML_ASSERT(false); // TODO: implement
    }
}

static void ggml_compute_forward_dup_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
        // parallelize by elements
        const int ne = ggml_nelements(dst);
        const int dr = (ne + nth - 1) / nth;
        const int ie0 = dr * ith;
        const int ie1 = MIN(ie0 + dr, ne);

        memcpy(
            ((char *)  dst->data + ie0*nb0),
            ((char *) src0->data + ie0*nb00),
            (ie1 - ie0) * GGML_TYPE_SIZE[src0->type]);

        return;
    }

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == GGML_TYPE_SIZE[src0->type] && nb0 == GGML_TYPE_SIZE[dst->type]) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (ggml_is_contiguous(dst)) {
        // TODO: simplify
        if (nb00 == sizeof(float)) {
            if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = GGML_FP32_TO_FP16(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (ggml_is_quantized(dst->type)) {
                quantize_row_q_t const quantize_row_q = quantize_fns[dst->type].quantize_row_q;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / GGML_BLCK_SIZE[dst->type]);
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            quantize_row_q(src0_ptr, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                GGML_ASSERT(false); // TODO: implement
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == GGML_TYPE_F16) {
                size_t id = 0;
                ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = GGML_FP32_TO_FP16(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else {
                GGML_ASSERT(false); // TODO: implement
            }
        }

        return;
    }

    // dst counters

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(float));

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(ggml_fp16_t *) dst_ptr = GGML_FP32_TO_FP16(*(const float *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        GGML_ASSERT(false); // TODO: implement
    }
}

static void ggml_compute_forward_dup(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_dup_f16(params, src0, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_dup_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_add

static void ggml_compute_forward_add_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        for (int j = ith; j < n; j += nth) {
#ifdef GGML_USE_ACCELERATE
            vDSP_vadd(
                    (float *) ((char *) src0->data + j*nb01), 1,
                    (float *) ((char *) src1->data + j*nb11), 1,
                    (float *) ((char *) dst->data  + j*nb1),  1, nc);
#else
            ggml_vec_add_f32(nc,
                    (float *) ((char *) dst->data  + j*nb1),
                    (float *) ((char *) src0->data + j*nb01),
                    (float *) ((char *) src1->data + j*nb11));
#endif
        }
    } else {
        // src1 is not contiguous
        for (int j = ith; j < n; j += nth) {
            float * dst_ptr  = (float *) ((char *) dst->data  + j*nb1);
            float * src0_ptr = (float *) ((char *) src0->data + j*nb01);
            for (int i = 0; i < nc; i++) {
                float * src1_ptr = (float *) ((char *) src1->data + j*nb11 + i*nb10);

                dst_ptr[i] = src0_ptr[i] + *src1_ptr;
            }
        }
    }
}

static void ggml_compute_forward_add_f16_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

    if (nb10 == sizeof(float)) {
        for (int j = ith; j < n; j += nth) {
            ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *) dst->data  + j*nb1);
            ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + j*nb01);
            for (int i = 0; i < nc; i++) {
                float * src1_ptr = (float *) ((char *) src1->data + j*nb11 + i*nb10);
                dst_ptr[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src0_ptr[i]) + *src1_ptr);
            }
        }
    }
    else {
        // src1 is not contiguous
        GGML_ASSERT(false);
    }
}

static void ggml_compute_forward_add_f16_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

    if (nb10 == sizeof(ggml_fp16_t)) {
        for (int j = ith; j < n; j += nth) {
            ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *) dst->data  + j*nb1);
            ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + j*nb01);
            for (int i = 0; i < nc; i++) {
                ggml_fp16_t * src1_ptr = (ggml_fp16_t *) ((char *) src1->data + j*nb11 + i*nb10);
                dst_ptr[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src0_ptr[i]) + GGML_FP16_TO_FP32(*src1_ptr));
            }
        }
    }
    else {
        // src1 is not contiguous
        GGML_ASSERT(false);
    }
}

static void ggml_compute_forward_add_q_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    //const int64_t ne10 = src1->ne[0];
    //const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    //const int64_t ne0  = dst->ne[0];
    //const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];
    const int64_t ne3  = dst->ne[3];

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    const enum ggml_type type = src0->type;
    dequantize_row_q_t const dequantize_row_q = quantize_fns[type].dequantize_row_q;
    quantize_row_q_t const quantize_row_q = quantize_fns[type].quantize_row_q;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == (int) GGML_TYPE_SIZE[type]);
    GGML_ASSERT(nb10 == sizeof(float));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // total rows in src0
    const int nr = ne01*ne02*ne03;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        const int i13 = i03;
        const int i12 = i02;
        const int i11 = i01;

        const int i3 = i03;
        const int i2 = i02;
        const int i1 = i01;

        void  * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        float * src1_row = (float *)((char *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        void  * dst_row  = (void *) ((char *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb0));

        assert(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne00);
    }
}

static void ggml_compute_forward_add(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_add_f32(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F16:
            {
                if (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add_f16_f16(params, src0, src1, dst);
                }
                else if (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add_f16_f32(params, src0, src1, dst);
                }
                else {
                    GGML_ASSERT(false);
                }
            } break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_2:
        case GGML_TYPE_Q4_3:
            {
                ggml_compute_forward_add_q_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sub

static void ggml_compute_forward_sub_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_sub_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void ggml_compute_forward_sub(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_sub_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_mul

static void ggml_compute_forward_mul_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_mul_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void ggml_compute_forward_mul(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_mul_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_div

static void ggml_compute_forward_div_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_div_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void ggml_compute_forward_div(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_div_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sqr

static void ggml_compute_forward_sqr_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n     = ggml_nrows(src0);
    const int nc    = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_sqr_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_sqr(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_sqr_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sqrt

static void ggml_compute_forward_sqrt_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_sqrt_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_sqrt(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_sqrt_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sum

static void ggml_compute_forward_sum_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_is_scalar(dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    assert(ggml_is_scalar(dst));
    assert(src0->nb[0] == sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (float *) (dst->data),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
            }
        }
    }
}

static void ggml_compute_forward_sum(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_mean

static void ggml_compute_forward_mean_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    assert(ne0 == 1);
    assert(ne1 == ne01);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    UNUSED(ne0);
    UNUSED(ne1);
    UNUSED(ne2);
    UNUSED(ne3);

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (float *) ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (float) ne00;
            }
        }
    }
}

static void ggml_compute_forward_mean(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_repeat

static void ggml_compute_forward_repeat_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_can_repeat(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // TODO: implement support for rank > 2 tensors
    assert(src0->ne[2] == 1);
    assert(src0->ne[3] == 1);
    assert( dst->ne[2] == 1);
    assert( dst->ne[3] == 1);

    const int nc  = dst->ne[0];
    const int nr  = dst->ne[1];
    const int nc0 = src0->ne[0];
    const int nr0 = src0->ne[1];
    const int ncr = nc/nc0; // guaranteed to be an integer due to the check in ggml_can_repeat
    const int nrr = nr/nr0; // guaranteed to be an integer due to the check in ggml_can_repeat

    // TODO: support for transposed / permuted tensors
    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    // TODO: maybe this is not optimal?
    for (int i = 0; i < nrr; i++) {
        for (int j = 0; j < ncr; j++) {
            for (int k = 0; k < nr0; k++) {
                ggml_vec_cpy_f32(nc0,
                        (float *) ((char *)  dst->data + (i*nr0 + k)*( dst->nb[1]) + j*nc0*( dst->nb[0])),
                        (float *) ((char *) src0->data + (        k)*(src0->nb[1])));
            }
        }
    }
}

static void ggml_compute_forward_repeat(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_abs

static void ggml_compute_forward_abs_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_abs_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_abs(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_abs_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sgn

static void ggml_compute_forward_sgn_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_sgn_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_sgn(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_sgn_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_neg

static void ggml_compute_forward_neg_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_neg_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_neg(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_neg_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_step

static void ggml_compute_forward_step_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_step_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_step(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_step_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_relu

static void ggml_compute_forward_relu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_relu(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_relu_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_gelu

static void ggml_compute_forward_gelu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_gelu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void ggml_compute_forward_gelu(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    //printf("XXXXXXXX gelu\n");
}

// ggml_compute_forward_silu

static void ggml_compute_forward_silu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void ggml_compute_forward_silu(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}


// ggml_compute_forward_norm

static void ggml_compute_forward_norm_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const float eps = 1e-5f; // TODO: make this a parameter

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)x[i00];
                }

                float mean = sum/ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                ggml_float sum2 = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    float v = x[i00] - mean;
                    y[i00] = v;
                    sum2 += (ggml_float)(v*v);
                }

                float variance = sum2/ne00;
                const float scale = 1.0f/sqrtf(variance + eps);

                ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void ggml_compute_forward_norm(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

static void ggml_compute_forward_rms_norm_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const float eps = 1e-6f; // TODO: make this a parameter

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                float mean = sum/ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                memcpy(y, x, ne00 * sizeof(float));
                // for (int i00 = 0; i00 < ne00; i00++) {
                //     y[i00] = x[i00];
                // }

                const float scale = 1.0f/sqrtf(mean + eps);

                ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void ggml_compute_forward_rms_norm(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}


// ggml_compute_forward_mul_mat

#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CUBLAS)
// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool ggml_compute_forward_mul_mat_use_blas(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    //const int64_t ne00 = src0->ne[0];
    //const int64_t ne01 = src0->ne[1];

    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if (ggml_is_contiguous(src0) &&
        ggml_is_contiguous(src1) && ((ne0 >= 32 && ne1 >= 32 && ne10 >= 32))) {

        /*printf("BLAS: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
        return true;
    }

    return false;
}
#endif

static void ggml_compute_forward_mul_mat_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CUBLAS)
    const int64_t ne10 = src1->ne[0];
#endif
    const int64_t ne11 = src1->ne[1];
#ifndef NDEBUG
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];
    const int64_t ne3  = dst->ne[3];

    const int nb00 = src0->nb[0];
#endif
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

#ifndef NDEBUG
    const int nb10 = src1->nb[0];
#endif
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    assert(ne02 == ne12);
    assert(ne03 == ne13);
    assert(ne2  == ne12);
    assert(ne3  == ne13);

    // we don't support permuted src0 or src1
    assert(nb00 == sizeof(float));
    assert(nb10 == sizeof(float));

    // dst cannot be transposed or permuted
    assert(nb0 == sizeof(float));
    assert(nb0 <= nb1);
    assert(nb1 <= nb2);
    assert(nb2 <= nb3);

    assert(ne0 == ne01);
    assert(ne1 == ne11);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CUBLAS)
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        if (params->ith != 0) {
            return;
        }

        if (params->type == GGML_TASK_INIT) {
            return;
        }

        if (params->type == GGML_TASK_FINALIZE) {
            return;
        }

#if defined(GGML_USE_CUBLAS)
        float *d_X = NULL;
        float *d_Y = NULL;
        float *d_D = NULL;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int x_ne = ne01 * ne10;
        const int y_ne = ne11 * ne10;
        const int d_ne = ne11 * ne01;

        CUDA_CHECK(cudaMalloc((void **)(&d_X), sizeof(float) * x_ne));
        CUDA_CHECK(cudaMalloc((void **)(&d_Y), sizeof(float) * y_ne));
        CUDA_CHECK(cudaMalloc((void **)(&d_D), sizeof(float) * d_ne));
#endif

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

#if defined(GGML_USE_CUBLAS)
                // copy data to device
                CUDA_CHECK(cudaMemcpyAsync(d_X, x, sizeof(float) * x_ne, cudaMemcpyHostToDevice, cudaStream));
                CUDA_CHECK(cudaMemcpyAsync(d_Y, y, sizeof(float) * y_ne, cudaMemcpyHostToDevice, cudaStream));

                // compute
                CUBLAS_CHECK(
                    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            &alpha, d_X, ne00,
                                    d_Y, ne10,
                            &beta,  d_D, ne01));

                // copy data to host
                CUDA_CHECK(cudaMemcpyAsync(d, d_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, cudaStream));
#else
                // zT = y * xT
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne00,
                        0.0f,    d, ne01);
#endif
            }
        }
#if defined(GGML_USE_CUBLAS)
        CUDA_CHECK(cudaStreamSynchronize(cudaStream));
        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_Y));
        CUDA_CHECK(cudaFree(d_D));
#endif
        //printf("CBLAS F32 = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

        return;
    }
#endif

    if (params->type == GGML_TASK_INIT) {
        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // parallelize by src0 rows using ggml_vec_dot_f32

    // total rows in src0
    const int nr = ne01*ne02*ne03;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        for (int64_t ic = 0; ic < ne11; ++ic) {
            // src1 indices
            const int i13 = i03;
            const int i12 = i02;
            const int i11 = ic;

            // dst indices
            const int i0 = i01;
            const int i1 = i11;
            const int i2 = i02;
            const int i3 = i03;

            ggml_vec_dot_f32(ne00,
                    (float *) ((char *)  dst->data + (i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3)),
                    (float *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03)),
                    (float *) ((char *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13)));
        }
    }

    //int64_t t1 = ggml_perf_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);
    //    printf("nb10 = %5d, nb11 = %5d, nb12 = %5d, nb13 = %5d\n", nb10, nb11, nb12, nb13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}

static void ggml_compute_forward_mul_mat_f16_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];
    const int64_t ne3  = dst->ne[3];
    //const int64_t ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // TODO: we don't support permuted src0
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CUBLAS)
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        GGML_ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == GGML_TASK_INIT) {
            return;
        }

        if (params->type == GGML_TASK_FINALIZE) {
            return;
        }

#if defined(GGML_USE_CUBLAS)
        ggml_fp16_t * const wdata = params->wdata;

        float *d_X = NULL;
        float *d_Y = NULL;
        float *d_D = NULL;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int x_ne = ne01 * ne10;
        const int y_ne = ne11 * ne10;
        const int d_ne = ne11 * ne01;

        CUDA_CHECK(cudaMalloc((void **)(&d_X), sizeof(ggml_fp16_t) * x_ne));
        CUDA_CHECK(cudaMalloc((void **)(&d_Y), sizeof(float) * y_ne));
        CUDA_CHECK(cudaMalloc((void **)(&d_D), sizeof(float) * d_ne));
#else
        float * const wdata = params->wdata;
#endif
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
#if defined(GGML_USE_CUBLAS)
                // with cuBlAS, instead of converting src0 to fp32, we convert src1 to fp16
                {
                    size_t id = 0;
                    for (int64_t i01 = 0; i01 < ne11; ++i01) {
                        for (int64_t i00 = 0; i00 < ne10; ++i00) {
                            wdata[id++] = GGML_FP32_TO_FP16(*(float *) ((char *) src1->data + i03*nb13 + i02*nb12 + i01*nb11 + i00*nb10));
                        }
                    }
                }
#else
                {
                    size_t id = 0;
                    for (int64_t i01 = 0; i01 < ne01; ++i01) {
                        for (int64_t i00 = 0; i00 < ne00; ++i00) {
                            wdata[id++] = GGML_FP16_TO_FP32(*(ggml_fp16_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00));
                        }
                    }
                }
#endif

#if defined(GGML_USE_CUBLAS)
                const ggml_fp16_t * x = (ggml_fp16_t *) ((char *) src0->data + i02*nb02 + i03*nb03);
                const ggml_fp16_t * y = (ggml_fp16_t *) wdata;

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // copy data to device
                CUDA_CHECK(cudaMemcpyAsync(d_X, x, sizeof(ggml_fp16_t) * x_ne, cudaMemcpyHostToDevice, cudaStream));
                CUDA_CHECK(cudaMemcpyAsync(d_Y, y, sizeof(ggml_fp16_t) * y_ne, cudaMemcpyHostToDevice, cudaStream));

                // compute
                CUBLAS_CHECK(
                    cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            &alpha, d_X, CUDA_R_16F, ne00,
                                    d_Y, CUDA_R_16F, ne10,
                            &beta,  d_D, CUDA_R_32F, ne01,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));

                // copy data to host
                CUDA_CHECK(cudaMemcpyAsync(d, d_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, cudaStream));
#else
                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // zT = y * xT
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne00,
                        0.0f,    d, ne01);
#endif
            }
        }

#if defined(GGML_USE_CUBLAS)
        CUDA_CHECK(cudaStreamSynchronize(cudaStream));
        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_Y));
        CUDA_CHECK(cudaFree(d_D));
#endif
        /*printf("CBLAS F16 = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);*/

        return;
    }
#endif

    if (params->type == GGML_TASK_INIT) {
        ggml_fp16_t * const wdata = params->wdata;

        size_t id = 0;
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    for (int64_t i10 = 0; i10 < ne10; ++i10) {
                        wdata[id++] = GGML_FP32_TO_FP16(*(float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10));
                    }
                }
            }
        }

        GGML_ASSERT(id*sizeof(ggml_fp16_t) <= params->wsize);

        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // fp16 -> half the size, so divide by 2
    // TODO: do not support transposed src1
    assert(nb10/2 == sizeof(ggml_fp16_t));

    // parallelize by src0 rows using ggml_vec_dot_f16

    // total rows in src0
    const int nr = ne01*ne02*ne03;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * wdata = params->wdata;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int i13 = i03;
        const int i12 = i02;

        const int i0 = i01;
        const int i2 = i02;
        const int i3 = i03;

        ggml_fp16_t * src0_row = (ggml_fp16_t *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        ggml_fp16_t * src1_col =                                wdata + (       0 + i12*ne11 + i13*ne12*ne11)*ne00;

        float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

        for (int64_t ic = 0; ic < ne11; ++ic) {
            ggml_vec_dot_f16(ne00, &dst_col[ic*ne0], src0_row, src1_col + ic*ne00);
        }
    }

    //int64_t t1 = ggml_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}

static void ggml_compute_forward_mul_mat_q_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];
    const int64_t ne3  = dst->ne[3];

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    const enum ggml_type type = src0->type;
    quantize_row_q_t const quantize_row_q_dot = quantize_fns[type].quantize_row_q_dot;
    vec_dot_q_t      const vec_dot_q          = quantize_fns[type].vec_dot_q;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == (int) GGML_TYPE_SIZE[type]);
    GGML_ASSERT(nb10 == sizeof(float));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CUBLAS)
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        if (params->ith != 0) {
            return;
        }

        if (params->type == GGML_TASK_INIT) {
            return;
        }

        if (params->type == GGML_TASK_FINALIZE) {
            return;
        }

#if defined(GGML_USE_CUBLAS)
        float *d_X = NULL;
        float *d_Y = NULL;
        float *d_D = NULL;
        float *d_Q = NULL;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int x_ne = ne01 * ne10;
        const int y_ne = ne11 * ne10;
        const int d_ne = ne11 * ne01;

        CUDA_CHECK(cudaMalloc((void **)(&d_X), sizeof(float) * x_ne));
        CUDA_CHECK(cudaMalloc((void **)(&d_Y), sizeof(float) * y_ne));
        CUDA_CHECK(cudaMalloc((void **)(&d_D), sizeof(float) * d_ne));
        CUDA_CHECK(cudaMalloc((void **)(&d_Q), GGML_TYPE_SIZE[type] * x_ne / GGML_BLCK_SIZE[type]));

        void (*dequantize_row_q_cuda)(const void * x, float * y, int k, cudaStream_t stream)  = NULL;
        if (type == GGML_TYPE_Q4_0) {
            dequantize_row_q_cuda = dequantize_row_q4_0_cuda;
        }
        else if (type == GGML_TYPE_Q4_1) {
            dequantize_row_q_cuda = dequantize_row_q4_1_cuda;
        }
        else if (type == GGML_TYPE_Q4_2) {
            dequantize_row_q_cuda = dequantize_row_q4_2_cuda;
        }
        else {
            GGML_ASSERT(false);
        }
#else
        float * const wdata = params->wdata;
        dequantize_row_q_t const dequantize_row_q = quantize_fns[type].dequantize_row_q;
#endif

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

#if defined(GGML_USE_CUBLAS)
                // copy and dequantize on device
                CUDA_CHECK(
                    cudaMemcpyAsync(d_Q, (char *) src0->data + i03*nb03 + i02*nb02,
                        GGML_TYPE_SIZE[type] * x_ne / GGML_BLCK_SIZE[type], cudaMemcpyHostToDevice, cudaStream));

                dequantize_row_q_cuda(d_Q, d_X, ne01 * ne00, cudaStream);
                CUDA_CHECK(cudaGetLastError());
#else
                {
                    size_t id = 0;
                    for (int64_t i01 = 0; i01 < ne01; ++i01) {
                        dequantize_row_q((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01, wdata + id, ne00);
                        id += ne00;
                    }
                }
                const float * x = wdata;
#endif


#if defined(GGML_USE_CUBLAS)
                // copy data to device
                CUDA_CHECK(cudaMemcpyAsync(d_Y, y, sizeof(float) * y_ne, cudaMemcpyHostToDevice, cudaStream));

                // compute
                CUBLAS_CHECK(
                    cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            &alpha, d_X, ne00,
                                    d_Y, ne10,
                            &beta,  d_D, ne01));

                // copy data to host
                CUDA_CHECK(cudaMemcpyAsync(d, d_D, sizeof(float) * d_ne, cudaMemcpyDeviceToHost, cudaStream));
#else
                // zT = y * xT
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne11, ne01, ne10,
                        1.0f,    y, ne10,
                                 x, ne00,
                        0.0f,    d, ne01);
#endif
            }
        }

#if defined(GGML_USE_CUBLAS)
        CUDA_CHECK(cudaStreamSynchronize(cudaStream));
        CUDA_CHECK(cudaFree(d_X));
        CUDA_CHECK(cudaFree(d_Y));
        CUDA_CHECK(cudaFree(d_D));
        CUDA_CHECK(cudaFree(d_Q));
#endif
        //printf("CBLAS = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

        return;
    }
#endif

    if (params->type == GGML_TASK_INIT) {
        char * wdata = params->wdata;
        const size_t row_size = ne10*GGML_TYPE_SIZE[GGML_TYPE_Q8_0]/GGML_BLCK_SIZE[GGML_TYPE_Q8_0];

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    quantize_row_q_dot((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) wdata, ne10);
                    wdata += row_size;
                }
            }
        }

        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // parallelize by src0 rows using ggml_vec_dot_q

    // total rows in src0
    const int nr = ne01*ne02*ne03;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    void * wdata = params->wdata;
    const size_t row_size = ne00*GGML_TYPE_SIZE[GGML_TYPE_Q8_0]/GGML_BLCK_SIZE[GGML_TYPE_Q8_0];

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int i13 = i03;
        const int i12 = i02;

        const int i0 = i01;
        const int i2 = i02;
        const int i3 = i03;

        void * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        char * src1_col =          ((char *)      wdata + (      (0 + i12*ne11 + i13*ne12*ne11)*row_size));

        float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

        assert(ne00 % 32 == 0);

        for (int64_t ic = 0; ic < ne11; ++ic) {
            vec_dot_q(ne00, &dst_col[ic*ne0], src0_row, (void *) (src1_col + ic*row_size));
        }
    }

    //int64_t t1 = ggml_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}

static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_2:
        case GGML_TYPE_Q4_3:
        case GGML_TYPE_Q8_0:
            {
                ggml_compute_forward_mul_mat_q_f32(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_mul_mat_f16_f32(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_mul_mat_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_scale

static void ggml_compute_forward_scale_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // scale factor
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), v);
    }
}

static void ggml_compute_forward_scale(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_cpy

static void ggml_compute_forward_cpy(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ggml_compute_forward_dup(params, src0, dst);
}

// ggml_compute_forward_cont

static void ggml_compute_forward_cont(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ggml_compute_forward_dup(params, src0, dst);
}

// ggml_compute_forward_reshape

static void ggml_compute_forward_reshape(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
    UNUSED(dst);
}

// ggml_compute_forward_view

static void ggml_compute_forward_view(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// ggml_compute_forward_permute

static void ggml_compute_forward_permute(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// ggml_compute_forward_transpose

static void ggml_compute_forward_transpose(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// ggml_compute_forward_get_rows

static void ggml_compute_forward_get_rows_q(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);
    const enum ggml_type type = src0->type;
    dequantize_row_q_t const dequantize_row_q = quantize_fns[type].dequantize_row_q;

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == GGML_TYPE_SIZE[type]);

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        dequantize_row_q(
                (const void *) ((char *) src0->data + r*src0->nb[1]),
                     (float *) ((char *)  dst->data + i*dst->nb[1]), nc);
    }
}

static void ggml_compute_forward_get_rows_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == sizeof(ggml_fp16_t));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        for (int j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((char *) src0->data + r*src0->nb[1]))[j];
            ((float *) ((char *)  dst->data + i*dst->nb[1]))[j] = GGML_FP16_TO_FP32(v);
        }
    }
}

static void ggml_compute_forward_get_rows_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i*dst->nb[1]),
                (float *) ((char *) src0->data + r*src0->nb[1]));
    }
}

static void ggml_compute_forward_get_rows(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_2:
        case GGML_TYPE_Q4_3:
        case GGML_TYPE_Q8_0:
            {
                ggml_compute_forward_get_rows_q(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// ggml_compute_forward_diag_mask_inf

static void ggml_compute_forward_diag_mask_inf_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(src1->type == GGML_TYPE_I32);
    assert(ggml_nelements(src1) == 1);

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];

    // TODO: handle transposed/permuted matrices

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];
    const int nr = src0->ne[1];
    const int nz = n/nr;

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < nr; j++) {
            for (int i = n_past; i < nc; i++) {
                if (i > n_past + j) {
                    *(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = -INFINITY;
                }
            }
        }
    }
}

static void ggml_compute_forward_diag_mask_inf(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_inf_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_soft_max

static void ggml_compute_forward_soft_max_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // TODO: handle transposed/permuted matrices

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float *p = (float *)((char *) dst->data + i1*dst->nb[1]);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(p[i]));
        }
#endif

        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, p);

        ggml_float sum = 0.0;

        uint16_t scvt;
        for (int i = 0; i < nc; i++) {
            if (p[i] == -INFINITY) {
                p[i] = 0.0f;
            } else {
                //const float val = (p[i] == -INFINITY) ? 0.0 : exp(p[i] - max);
                ggml_fp16_t s = GGML_FP32_TO_FP16(p[i] - max);
                memcpy(&scvt, &s, sizeof(scvt));
                const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt]);
                sum += (ggml_float)val;
                p[i] = val;
            }
        }

        assert(sum > 0.0);

        sum = 1.0/sum;
        ggml_vec_scale_f32(nc, p, sum);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(p[i]));
            assert(!isinf(p[i]));
        }
#endif
    }
}

static void ggml_compute_forward_soft_max(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, src0, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_rope

static void ggml_compute_forward_rope_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(src1->type == GGML_TYPE_I32);
    assert(ggml_nelements(src1) == 3);

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];
    const int n_dims = ((int32_t *) src1->data)[1];
    const int mode   = ((int32_t *) src1->data)[2];

    //const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const int nb0 = src0->nb[0];
    const int nb1 = src0->nb[1];
    const int nb2 = src0->nb[2];
    const int nb3 = src0->nb[3];

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    assert(nb0 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(10000.0, -2.0f/n_dims);

    const bool is_neox = mode & 2;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
            const int p = ((mode & 1) == 0 ? n_past + i2 : i2);
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                float theta = (float)p;

                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float cos_theta = cosf(theta);
                    const float sin_theta = sinf(theta);

                    theta *= theta_scale;

                    if (!is_neox) {
                        const float * const src = (float *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

                        const float x0 = src[0];
                        const float x1 = src[1];

                        dst_data[0] = x0*cos_theta - x1*sin_theta;
                        dst_data[1] = x0*sin_theta + x1*cos_theta;
                    } else {
                        const float * const src = (float *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + (i0/2)*nb0);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (i0/2)*nb0);

                        const float x0 = src[0];
                        const float x1 = src[n_dims/2];

                        dst_data[0]        = x0*cos_theta - x1*sin_theta;
                        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_rope_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(src1->type == GGML_TYPE_I32);
    assert(ggml_nelements(src1) == 3);

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];
    const int n_dims = ((int32_t *) src1->data)[1];
    const int mode   = ((int32_t *) src1->data)[2];

    //const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const int nb0 = src0->nb[0];
    const int nb1 = src0->nb[1];
    const int nb2 = src0->nb[2];
    const int nb3 = src0->nb[3];

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    assert(nb0 == sizeof(ggml_fp16_t));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(10000.0, -2.0f/n_dims);

    const bool is_neox = mode & 2;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = ((mode & 1) == 0 ? 0 : n_past); i2 < ne2; i2++) {
            const int p = ((mode & 1) == 0 ? n_past + i2 : i2);
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                float theta = (float)p;

                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const float cos_theta = cosf(theta);
                    const float sin_theta = sinf(theta);

                    theta *= theta_scale;

                    if (!is_neox) {
                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                              ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

                        const float x0 = GGML_FP16_TO_FP32(src[0]);
                        const float x1 = GGML_FP16_TO_FP32(src[1]);

                        dst_data[0] = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[1] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    } else {
                        const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + (i0/2)*nb0);
                              ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (i0/2)*nb0);

                        const float x0 = GGML_FP16_TO_FP32(src[0]);
                        const float x1 = GGML_FP16_TO_FP32(src[n_dims/2]);

                        dst_data[0]        = GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[n_dims/2] = GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                }
            }
        }
    }
}

static void ggml_compute_forward_rope(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_f16(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_conv_1d_1s

static void ggml_compute_forward_conv_1d_1s_f16_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    //const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    //const int64_t ne12 = src1->ne[2];
    //const int64_t ne13 = src1->ne[3];

    //const int64_t ne0  = dst->ne[0];
    //const int64_t ne1  = dst->ne[1];
    //const int64_t ne2  = dst->ne[2];
    //const int64_t ne3  = dst->ne[3];
    //const int64_t ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = ggml_up32(ne01);

    GGML_ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb10 == sizeof(float));

    if (params->type == GGML_TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i02*ew0*ne00;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + ne02*ew0*ne00;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                ggml_fp16_t * dst_data = wdata;
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = GGML_FP32_TO_FP16(src[i10]);
                }
            }
        }

        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int64_t i0 = 0; i0 < ne10; ++i0) {
            dst_data[i0] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                ggml_vec_dot_f16(ew0, &v,
                        (ggml_fp16_t *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (ggml_fp16_t *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0] += v;
            }
        }
    }
}

static void ggml_compute_forward_conv_1d_1s_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    //const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    //const int64_t ne12 = src1->ne[2];
    //const int64_t ne13 = src1->ne[3];

    //const int64_t ne0  = dst->ne[0];
    //const int64_t ne1  = dst->ne[1];
    //const int64_t ne2  = dst->ne[2];
    //const int64_t ne3  = dst->ne[3];
    //const int64_t ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = ggml_up32(ne01);

    GGML_ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));

    if (params->type == GGML_TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i02*ew0*ne00;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + ne02*ew0*ne00;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                float * dst_data = wdata;
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = src[i10];
                }
            }
        }

        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int64_t i0 = 0; i0 < ne10; ++i0) {
            dst_data[i0] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                ggml_vec_dot_f32(ew0, &v,
                        (float *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (float *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0] += v;
            }
        }
    }
}

static void ggml_compute_forward_conv_1d_1s(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_1d_1s_f16_f32(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_1d_1s_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_conv_1d_2s

static void ggml_compute_forward_conv_1d_2s_f16_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    //const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    //const int64_t ne12 = src1->ne[2];
    //const int64_t ne13 = src1->ne[3];

    //const int64_t ne0  = dst->ne[0];
    //const int64_t ne1  = dst->ne[1];
    //const int64_t ne2  = dst->ne[2];
    //const int64_t ne3  = dst->ne[3];
    //const int64_t ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = ggml_up32(ne01);

    GGML_ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb10 == sizeof(float));

    if (params->type == GGML_TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i02*ew0*ne00;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + ne02*ew0*ne00;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                ggml_fp16_t * dst_data = wdata;
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = GGML_FP32_TO_FP16(src[i10]);
                }
            }
        }

        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
            dst_data[i0/2] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                ggml_vec_dot_f16(ew0, &v,
                        (ggml_fp16_t *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (ggml_fp16_t *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0/2] += v;
            }
        }
    }
}

static void ggml_compute_forward_conv_1d_2s_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    //const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    //const int64_t ne12 = src1->ne[2];
    //const int64_t ne13 = src1->ne[3];

    //const int64_t ne0  = dst->ne[0];
    //const int64_t ne1  = dst->ne[1];
    //const int64_t ne2  = dst->ne[2];
    //const int64_t ne3  = dst->ne[3];
    //const int64_t ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = ggml_up32(ne01);

    GGML_ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));

    if (params->type == GGML_TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i02*ew0*ne00;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + ne02*ew0*ne00;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                float * dst_data = wdata;
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = src[i10];
                }
            }
        }

        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
            dst_data[i0/2] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                ggml_vec_dot_f32(ew0, &v,
                        (float *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (float *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0/2] += v;
            }
        }
    }
}

static void ggml_compute_forward_conv_1d_2s(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_1d_2s_f16_f32(params, src0, src1, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_1d_2s_f32(params, src0, src1, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_flash_attn

static void ggml_compute_forward_flash_attn_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        const bool masked,
             struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t neq0 = q->ne[0];
    const int64_t neq1 = q->ne[1];
    const int64_t neq2 = q->ne[2];
    const int64_t neq3 = q->ne[3];

    const int64_t nek0 = k->ne[0];
    const int64_t nek1 = k->ne[1];
    //const int64_t nek2 = k->ne[2];
    //const int64_t nek3 = k->ne[3];

    //const int64_t nev0 = v->ne[0];
    const int64_t nev1 = v->ne[1];
    //const int64_t nev2 = v->ne[2];
    //const int64_t nev3 = v->ne[3];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    //const int64_t ne2  = dst->ne[2];
    //const int64_t ne3  = dst->ne[3];

    const int nbk0 = k->nb[0];
    const int nbk1 = k->nb[1];
    const int nbk2 = k->nb[2];
    const int nbk3 = k->nb[3];

    const int nbq0 = q->nb[0];
    const int nbq1 = q->nb[1];
    const int nbq2 = q->nb[2];
    const int nbq3 = q->nb[3];

    const int nbv0 = v->nb[0];
    const int nbv1 = v->nb[1];
    const int nbv2 = v->nb[2];
    const int nbv3 = v->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t D = neq0;
    const int64_t N = neq1;
    const int64_t P = nek1 - N;
    const int64_t M = P + N;

    const int Mup = ggml_up(M, GGML_SOFT_MAX_UNROLL);

    GGML_ASSERT(ne0 == D);
    GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == sizeof(float));
    GGML_ASSERT(nbk0 == sizeof(float));
    GGML_ASSERT(nbv0 == sizeof(float));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    if (params->type == GGML_TASK_INIT) {
        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // parallelize by q rows using ggml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        float * S = (float *) params->wdata + ith*(Mup + CACHE_LINE_SIZE_F32);

        for (int i = M; i < Mup; ++i) {
            S[i] = -INFINITY;
        }

        for (int64_t ic = 0; ic < nek1; ++ic) {
            // k indices
            const int ik3 = iq3;
            const int ik2 = iq2;
            const int ik1 = ic;

            // S indices
            const int i1 = ik1;

            ggml_vec_dot_f32(neq0,
                    S + i1,
                    (float *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                    (float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
        }

        // scale
        ggml_vec_scale_f32(nek1, S, scale);

        if (masked) {
            for (int64_t i = P; i < M; i++) {
                if (i > P + iq1) {
                    S[i] = -INFINITY;
                }
            }
        }

        // softmax
        {
            float max = -INFINITY;
            ggml_vec_max_f32(M, &max, S);

            ggml_float sum = 0.0;
            {
#ifdef GGML_SOFT_MAX_ACCELERATE
                max = -max;
                vDSP_vsadd(S, 1, &max, S, 1, Mup);
                vvexpf(S, S, &Mup);
                ggml_vec_sum_f32(Mup, &sum, S);
#else
                uint16_t   scvt[GGML_SOFT_MAX_UNROLL];
                ggml_float sump[GGML_SOFT_MAX_UNROLL] = { 0.0 };

                for (int i = 0; i < Mup; i += GGML_SOFT_MAX_UNROLL) {
                    float * SS = S + i;

                    for (int j = 0; j < GGML_SOFT_MAX_UNROLL; ++j) {
                        if (SS[j] == -INFINITY) {
                            SS[j] = 0.0f;
                        } else {
                            ggml_fp16_t s = GGML_FP32_TO_FP16(SS[j] - max);
                            memcpy(&scvt[j], &s, sizeof(uint16_t));
                            const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt[j]]);
                            sump[j] += (ggml_float)val;
                            SS[j] = val;
                        }
                    }
                }

                for (int i = 0; i < GGML_SOFT_MAX_UNROLL; i++) {
                    sum += sump[i];
                }
#endif
            }

            assert(sum > 0.0);

            sum = 1.0/sum;
            ggml_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
            for (int i = 0; i < M; ++i) {
                assert(!isnan(S[i]));
                assert(!isinf(S[i]));
            }
#endif
        }

        for (int64_t ic = 0; ic < nev1; ++ic) {
            // dst indices
            const int i1 = iq1;
            const int i2 = iq2;
            const int i3 = iq3;

            ggml_vec_dot_f32(nek1,
                    (float *) ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                    (float *) ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                    S);
        }
    }
}

static void ggml_compute_forward_flash_attn_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        const bool masked,
             struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t neq0 = q->ne[0];
    const int64_t neq1 = q->ne[1];
    const int64_t neq2 = q->ne[2];
    const int64_t neq3 = q->ne[3];

    const int64_t nek0 = k->ne[0];
    const int64_t nek1 = k->ne[1];
    //const int64_t nek2 = k->ne[2];
    //const int64_t nek3 = k->ne[3];

    //const int64_t nev0 = v->ne[0];
    const int64_t nev1 = v->ne[1];
    //const int64_t nev2 = v->ne[2];
    //const int64_t nev3 = v->ne[3];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    //const int64_t ne2  = dst->ne[2];
    //const int64_t ne3  = dst->ne[3];

    const int nbk0 = k->nb[0];
    const int nbk1 = k->nb[1];
    const int nbk2 = k->nb[2];
    const int nbk3 = k->nb[3];

    const int nbq0 = q->nb[0];
    const int nbq1 = q->nb[1];
    const int nbq2 = q->nb[2];
    const int nbq3 = q->nb[3];

    const int nbv0 = v->nb[0];
    const int nbv1 = v->nb[1];
    const int nbv2 = v->nb[2];
    const int nbv3 = v->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t D = neq0;
    const int64_t N = neq1;
    const int64_t P = nek1 - N;
    const int64_t M = P + N;

    const int Mup = ggml_up(M, GGML_SOFT_MAX_UNROLL);

    GGML_ASSERT(ne0 == D);
    GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nbk0 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nbv0 == sizeof(ggml_fp16_t));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    if (params->type == GGML_TASK_INIT) {
        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // parallelize by q rows using ggml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        float * S = (float *) params->wdata + ith*(2*Mup + CACHE_LINE_SIZE_F32);

        for (int i = M; i < Mup; ++i) {
            S[i] = -INFINITY;
        }

        if (GGML_VEC_DOT_UNROLL > 2 || nek1 % GGML_VEC_DOT_UNROLL != 0) {
            for (int64_t ic = 0; ic < nek1; ++ic) {
                // k indices
                const int ik3 = iq3;
                const int ik2 = iq2;
                const int ik1 = ic;

                // S indices
                const int i1 = ik1;

                ggml_vec_dot_f16(neq0,
                        S + i1,
                        (ggml_fp16_t *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                        (ggml_fp16_t *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
            }
        } else {
            for (int64_t ic = 0; ic < nek1; ic += GGML_VEC_DOT_UNROLL) {
                // k indices
                const int ik3 = iq3;
                const int ik2 = iq2;
                const int ik1 = ic;

                // S indices
                const int i1 = ik1;

                ggml_vec_dot_f16_unroll(neq0, nbk1,
                        S + i1,
                        ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                        (ggml_fp16_t *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
            }
        }

        // scale
        ggml_vec_scale_f32(nek1, S, scale);

        if (masked) {
            for (int64_t i = P; i < M; i++) {
                if (i > P + iq1) {
                    S[i] = -INFINITY;
                }
            }
        }

        // softmax
        {
            float max = -INFINITY;
            ggml_vec_max_f32(M, &max, S);

            ggml_float sum = 0.0;
            {
#ifdef GGML_SOFT_MAX_ACCELERATE
                max = -max;
                vDSP_vsadd(S, 1, &max, S, 1, Mup);
                vvexpf(S, S, &Mup);
                ggml_vec_sum_f32(Mup, &sum, S);
#else
                uint16_t   scvt[GGML_SOFT_MAX_UNROLL];
                ggml_float sump[GGML_SOFT_MAX_UNROLL] = { 0.0 };

                for (int i = 0; i < Mup; i += GGML_SOFT_MAX_UNROLL) {
                    float * SS = S + i;

                    for (int j = 0; j < GGML_SOFT_MAX_UNROLL; ++j) {
                        if (SS[j] == -INFINITY) {
                            SS[j] = 0.0f;
                        } else {
                            ggml_fp16_t s = GGML_FP32_TO_FP16(SS[j] - max);
                            memcpy(&scvt[j], &s, sizeof(uint16_t));
                            const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt[j]]);
                            sump[j] += (ggml_float)val;
                            SS[j] = val;
                        }
                    }
                }

                for (int i = 0; i < GGML_SOFT_MAX_UNROLL; i++) {
                    sum += sump[i];
                }
#endif
            }

            assert(sum > 0.0);

            sum = 1.0/sum;
            ggml_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
            for (int i = 0; i < M; ++i) {
                assert(!isnan(S[i]));
                assert(!isinf(S[i]));
            }
#endif
        }

        ggml_fp16_t * S16 = (ggml_fp16_t *) ((float *) params->wdata + ith*(2*Mup + CACHE_LINE_SIZE_F32) + Mup);

        for (int64_t i = 0; i < M; i++) {
            S16[i] = GGML_FP32_TO_FP16(S[i]);
        }

        if (GGML_VEC_DOT_UNROLL == 1 || (nev1 % GGML_VEC_DOT_UNROLL != 0)) {
            for (int64_t ic = 0; ic < nev1; ++ic) {
                // dst indices
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;

                ggml_vec_dot_f16(nek1,
                        (float *)       ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                        (ggml_fp16_t *) ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                        S16);
            }
        } else {
            for (int64_t ic = 0; ic < nev1; ic += GGML_VEC_DOT_UNROLL) {
                // dst indices
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;

                ggml_vec_dot_f16_unroll(nek1, nbv1,
                        (float *) ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                        ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                        S16);
            }
        }
    }
}

static void ggml_compute_forward_flash_attn(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        const bool masked,
        struct ggml_tensor * dst) {
    switch (q->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_flash_attn_f16(params, q, k, v, masked, dst);
            } break;
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_f32(params, q, k, v, masked, dst);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_flash_ff

static void ggml_compute_forward_flash_ff_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * a,  // F16
        const struct ggml_tensor * b0, // F16 fc_w
        const struct ggml_tensor * b1, // F32 fc_b
        const struct ggml_tensor * c0, // F16 proj_w
        const struct ggml_tensor * c1, // F32 proj_b
        struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int64_t nea0 = a->ne[0];
    const int64_t nea1 = a->ne[1];
    const int64_t nea2 = a->ne[2];
    const int64_t nea3 = a->ne[3];

    const int64_t neb00 = b0->ne[0];
    const int64_t neb01 = b0->ne[1];
    //const int64_t neb02 = b0->ne[2];
    //const int64_t neb03 = b0->ne[3];

    const int64_t neb10 = b1->ne[0];
    const int64_t neb11 = b1->ne[1];
    //const int64_t neb12 = b1->ne[2];
    //const int64_t neb13 = b1->ne[3];

    const int64_t nec00 = c0->ne[0];
    const int64_t nec01 = c0->ne[1];
    //const int64_t nec02 = c0->ne[2];
    //const int64_t nec03 = c0->ne[3];

    const int64_t nec10 = c1->ne[0];
    const int64_t nec11 = c1->ne[1];
    //const int64_t nec12 = c1->ne[2];
    //const int64_t nec13 = c1->ne[3];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    //const int64_t ne3 = dst->ne[3];

    const int nba0 = a->nb[0];
    const int nba1 = a->nb[1];
    const int nba2 = a->nb[2];
    const int nba3 = a->nb[3];

    const int nbb00 = b0->nb[0];
    const int nbb01 = b0->nb[1];
    const int nbb02 = b0->nb[2];
    const int nbb03 = b0->nb[3];

    const int nbb10 = b1->nb[0];
    //const int nbb11 = b1->nb[1];
    //const int nbb12 = b1->nb[2];
    //const int nbb13 = b1->nb[3];

    const int nbc00 = c0->nb[0];
    const int nbc01 = c0->nb[1];
    const int nbc02 = c0->nb[2];
    const int nbc03 = c0->nb[3];

    const int nbc10 = c1->nb[0];
    //const int nbc11 = c1->nb[1];
    //const int nbc12 = c1->nb[2];
    //const int nbc13 = c1->nb[3];

    const int nb0 = dst->nb[0];
    const int nb1 = dst->nb[1];
    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t D = nea0;
    //const int64_t N = nea1;
    const int64_t M = neb01;

    GGML_ASSERT(ne0 == nea0);
    GGML_ASSERT(ne1 == nea1);
    GGML_ASSERT(ne2 == nea2);

    GGML_ASSERT(nba0  == sizeof(ggml_fp16_t));
    GGML_ASSERT(nbb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nbb10 == sizeof(float));
    GGML_ASSERT(nbc00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nbc10 == sizeof(float));

    GGML_ASSERT(neb00 == D);
    GGML_ASSERT(neb01 == M);
    GGML_ASSERT(neb10 == M);
    GGML_ASSERT(neb11 == 1);

    GGML_ASSERT(nec00 == M);
    GGML_ASSERT(nec01 == D);
    GGML_ASSERT(nec10 == D);
    GGML_ASSERT(nec11 == 1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    if (params->type == GGML_TASK_INIT) {
        return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // parallelize by a rows using ggml_vec_dot_f32

    // total rows in a
    const int nr = nea1*nea2*nea3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // a indices
        const int ia3 = ir/(nea2*nea1);
        const int ia2 = (ir - ia3*nea2*nea1)/nea1;
        const int ia1 = (ir - ia3*nea2*nea1 - ia2*nea1);

        float * S = (float *) params->wdata + ith*(2*M + CACHE_LINE_SIZE_F32);

        for (int64_t ic = 0; ic < neb01; ++ic) {
            // b0 indices
            const int ib03 = ia3;
            const int ib02 = ia2;
            const int ib01 = ic;

            // S indices
            const int i1 = ib01;

            ggml_vec_dot_f16(nea0,
                    S + i1,
                    (ggml_fp16_t *) ((char *) b0->data + (ib01*nbb01 + ib02*nbb02 + ib03*nbb03)),
                    (ggml_fp16_t *) ((char *)  a->data + ( ia1*nba1  +  ia2*nba2  +  ia3*nba3)));
        }

        ggml_vec_add_f32(neb01, S, S, (float *) b1->data);
        //ggml_vec_gelu_f32(neb01, S, S);

        ggml_fp16_t * S16 = (ggml_fp16_t *) ((float *) params->wdata + ith*(2*M + CACHE_LINE_SIZE_F32) + M);

        for (int64_t i = 0; i < M; i++) {
            S16[i] = GGML_FP32_TO_FP16(S[i]);
        }

        ggml_vec_gelu_f16(neb01, S16, S16);

        {
            // dst indices
            const int i1 = ia1;
            const int i2 = ia2;
            const int i3 = ia3;

            for (int64_t ic = 0; ic < nec01; ++ic) {

                ggml_vec_dot_f16(neb01,
                        (float *)       ((char *) dst->data + (ic*nb0 + i1*nb1   + i2*nb2   + i3*nb3)),
                        (ggml_fp16_t *) ((char *) c0->data  + (         ic*nbc01 + i2*nbc02 + i3*nbc03)),
                        S16);
            }

            ggml_vec_add_f32(nec01,
                    (float *) ((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3)),
                    (float *) ((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3)),
                    (float *) c1->data);
        }
    }
}

static void ggml_compute_forward_flash_ff(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b0,
        const struct ggml_tensor * b1,
        const struct ggml_tensor * c0,
        const struct ggml_tensor * c1,
        struct ggml_tensor * dst) {
    switch (b0->type) {
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_flash_ff_f16(params, a, b0, b1, c0, c1, dst);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(false); // TODO
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_map_unary

static void ggml_compute_forward_map_unary_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst,
        const ggml_unary_op_f32_t fun) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        fun(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}


static void ggml_compute_forward_map_unary(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst,
        const ggml_unary_op_f32_t fun) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_map_unary_f32(params, src0, dst, fun);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_map_binary

static void ggml_compute_forward_map_binary_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst,
        const ggml_binary_op_f32_t fun) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        fun(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}


static void ggml_compute_forward_map_binary(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst,
        const ggml_binary_op_f32_t fun) {
    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_map_binary_f32(params, src0, src1, dst, fun);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor->src0, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor->src0, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor->src0, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor->src0, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor->src0, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor->src0, tensor);
            } break;
        case GGML_OP_ABS:
            {
                ggml_compute_forward_abs(params, tensor->src0, tensor);
            } break;
        case GGML_OP_SGN:
            {
                ggml_compute_forward_sgn(params, tensor->src0, tensor);
            } break;
        case GGML_OP_NEG:
            {
                ggml_compute_forward_neg(params, tensor->src0, tensor);
            } break;
        case GGML_OP_STEP:
            {
                ggml_compute_forward_step(params, tensor->src0, tensor);
            } break;
        case GGML_OP_RELU:
            {
                ggml_compute_forward_relu(params, tensor->src0, tensor);
            } break;
        case GGML_OP_GELU:
            {
                ggml_compute_forward_gelu(params, tensor->src0, tensor);
            } break;
        case GGML_OP_SILU:
            {
                ggml_compute_forward_silu(params, tensor->src0, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor->src0, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor->src0, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor->src0, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor->src0, tensor);
            } break;
        case GGML_OP_RESHAPE:
            {
                ggml_compute_forward_reshape(params, tensor->src0, tensor);
            } break;
        case GGML_OP_VIEW:
            {
                ggml_compute_forward_view(params, tensor->src0);
            } break;
        case GGML_OP_PERMUTE:
            {
                ggml_compute_forward_permute(params, tensor->src0);
            } break;
        case GGML_OP_TRANSPOSE:
            {
                ggml_compute_forward_transpose(params, tensor->src0);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor->src0, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_CONV_1D_1S:
            {
                ggml_compute_forward_conv_1d_1s(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_CONV_1D_2S:
            {
                ggml_compute_forward_conv_1d_2s(params, tensor->src0, tensor->src1, tensor);
            } break;
        case GGML_OP_FLASH_ATTN:
            {
                int32_t t = ggml_get_i32_1d(tensor->opt[1], 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn(params, tensor->src0, tensor->src1, tensor->opt[0], masked, tensor);
            } break;
        case GGML_OP_FLASH_FF:
            {
                ggml_compute_forward_flash_ff(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2], tensor);
            } break;
        case GGML_OP_MAP_UNARY:
            {
                const ggml_unary_op_f32_t fun = *((ggml_unary_op_f32_t *)tensor->opt[0]->data);
                ggml_compute_forward_map_unary(params, tensor->src0, tensor, fun);
            }
            break;
        case GGML_OP_MAP_BINARY:
            {
                const ggml_binary_op_f32_t fun = *((ggml_binary_op_f32_t *)tensor->opt[0]->data);
                ggml_compute_forward_map_binary(params, tensor->src0, tensor->src1, tensor, fun);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ASSERT(false);
            } break;
    }
}

////////////////////////////////////////////////////////////////////////////////

static void ggml_compute_backward(struct ggml_context * ctx, struct ggml_tensor * tensor, bool inplace) {
    struct ggml_tensor * src0 = tensor->src0;
    struct ggml_tensor * src1 = tensor->src1;

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
            } break;
        case GGML_OP_ADD:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
                if (src1->grad) {
                    src1->grad = ggml_add_impl(ctx, src1->grad, tensor->grad, inplace);
                }
            } break;
        case GGML_OP_SUB:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
                if (src1->grad) {
                    src1->grad = ggml_sub_impl(ctx, src1->grad, tensor->grad, inplace);
                }
            } break;
        case GGML_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_mul(ctx, src1, tensor->grad),
                                inplace);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_impl(ctx,
                                src1->grad,
                                ggml_mul(ctx, src0, tensor->grad),
                                inplace);
                }
            } break;
        case GGML_OP_DIV:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_div(ctx, tensor->grad, src1),
                                inplace);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_sub_impl(ctx,
                                src1->grad,
                                ggml_mul(ctx,
                                    tensor->grad,
                                    ggml_div(ctx, tensor, src1)),
                                inplace);
                }
            } break;
        case GGML_OP_SQR:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_mul(ctx,
                                    ggml_mul(ctx, src0, tensor->grad),
                                    ggml_repeat(ctx, ggml_new_f32(ctx, 2.0f), src0)),
                                inplace);
                }
            } break;
        case GGML_OP_SQRT:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_div(ctx,
                                    ggml_repeat(ctx, ggml_new_f32(ctx, 0.5f), tensor),
                                    tensor),
                                inplace);
                }
            } break;
        case GGML_OP_SUM:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_repeat(ctx, tensor->grad, src0->grad),
                                inplace);
                }
            } break;
        case GGML_OP_MEAN:
            {
                GGML_ASSERT(false); // TODO: implement
            } break;
        case GGML_OP_REPEAT:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_sum(ctx, tensor->grad),
                                inplace);
                }
            } break;
        case GGML_OP_ABS:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_mul(ctx,
                                    ggml_sgn(ctx, src0),
                                    tensor->grad),
                                inplace);
                }
            } break;
        case GGML_OP_SGN:
            {
                if (src0->grad) {
                    // noop
                }
            } break;
        case GGML_OP_NEG:
            {
                if (src0->grad) {
                    src0->grad = ggml_sub_impl(ctx, src0->grad, tensor->grad, inplace);
                }
            } break;
        case GGML_OP_STEP:
            {
                if (src0->grad) {
                    // noop
                }
            } break;
        case GGML_OP_RELU:
            {
                if (src0->grad) {
                    src0->grad = ggml_sub_impl(ctx,
                            src0->grad,
                            ggml_mul(ctx,
                                ggml_step(ctx, src0),
                                tensor->grad),
                            inplace);
                }
            } break;
        case GGML_OP_GELU:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_SILU:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_NORM:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_RMS_NORM:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_MUL_MAT:
            {
                if (src0->grad) {
                    // TODO: this requires outer product - ggml_out_prod(ctx, src1, tensor->grad);
                    GGML_ASSERT(false);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_impl(ctx,
                                src1->grad,
                                ggml_mul_mat(ctx,
                                    ggml_cont(ctx, ggml_transpose(ctx, src0)),
                                    tensor->grad),
                                inplace);
                }
            } break;
        case GGML_OP_SCALE:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_CPY:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_CONT:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_RESHAPE:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_VIEW:
            {
                GGML_ASSERT(false); // not supported
            } break;
        case GGML_OP_PERMUTE:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_TRANSPOSE:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_GET_ROWS:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_SOFT_MAX:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_ROPE:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_CONV_1D_1S:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_CONV_1D_2S:
            {
                GGML_ASSERT(false); // TODO: not implemented
            } break;
        case GGML_OP_FLASH_ATTN:
            {
                GGML_ASSERT(false); // not supported
            } break;
        case GGML_OP_FLASH_FF:
            {
                GGML_ASSERT(false); // not supported
            } break;
        case GGML_OP_MAP_UNARY:
        case GGML_OP_MAP_BINARY:
            {
                GGML_ASSERT(false); // not supported
            } break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ASSERT(false);
            } break;
    }
}

static void ggml_visit_parents(struct ggml_cgraph * cgraph, struct ggml_tensor * node) {
    if (node->grad == NULL) {
        // this usually happens when we generate intermediate nodes from constants in the backward pass
        // it can also happen during forward pass, if the user performs computations with constants
        if (node->op != GGML_OP_NONE) {
            //GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
        }
    }

    // check if already visited
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return;
        }
    }

    for (int i = 0; i < cgraph->n_leafs; i++) {
        if (cgraph->leafs[i] == node) {
            return;
        }
    }

    if (node->src0) {
        ggml_visit_parents(cgraph, node->src0);
    }

    if (node->src1) {
        ggml_visit_parents(cgraph, node->src1);
    }

    for (int i = 0; i < GGML_MAX_OPT; ++i) {
        if (node->opt[i]) {
            ggml_visit_parents(cgraph, node->opt[i]);
        }
    }

    if (node->op == GGML_OP_NONE && node->grad == NULL) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < GGML_MAX_NODES);

        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        GGML_ASSERT(cgraph->n_nodes < GGML_MAX_NODES);

        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->grads[cgraph->n_nodes] = node->grad;
        cgraph->n_nodes++;
    }
}

static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor, bool expand) {
    if (!expand) {
        cgraph->n_nodes = 0;
        cgraph->n_leafs = 0;
    }

    const int n0 = cgraph->n_nodes;
    UNUSED(n0);

    ggml_visit_parents(cgraph, tensor);

    const int n_new = cgraph->n_nodes - n0;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true);
}

struct ggml_cgraph ggml_build_forward(struct ggml_tensor * tensor) {
    struct ggml_cgraph result = {
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.n_threads    =*/ GGML_DEFAULT_N_THREADS,
        /*.work_size    =*/ 0,
        /*.work         =*/ NULL,
        /*.nodes        =*/ { NULL },
        /*.grads        =*/ { NULL },
        /*.leafs        =*/ { NULL },
        /*.perf_runs    =*/ 0,
        /*.perf_cycles  =*/ 0,
        /*.perf_time_us =*/ 0,
    };

    ggml_build_forward_impl(&result, tensor, false);

    return result;
}

struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep) {
    struct ggml_cgraph result = *gf;

    GGML_ASSERT(gf->n_nodes > 0);

    // if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
    if (keep) {
        for (int i = 0; i < gf->n_nodes; i++) {
            struct ggml_tensor * node = gf->nodes[i];

            if (node->grad) {
                node->grad = ggml_dup_tensor(ctx, node);
                gf->grads[i] = node->grad;
            }
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct ggml_tensor * node = gf->nodes[i];

        // because we detached the grad nodes from the original graph, we can afford inplace operations
        if (node->grad) {
            ggml_compute_backward(ctx, node, keep);
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct ggml_tensor * node = gf->nodes[i];

        if (node->is_param) {
            GGML_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
            ggml_build_forward_impl(&result, node->grad, true);
        }
    }

    return result;
}

//
// thread data
//
// synchronization is done via busy loops
// I tried using spin locks, but not sure how to use them correctly - the things I tried were slower than busy loops
//

#ifdef __APPLE__

//#include <os/lock.h>
//
//typedef os_unfair_lock ggml_lock_t;
//
//#define ggml_lock_init(x)    UNUSED(x)
//#define ggml_lock_destroy(x) UNUSED(x)
//#define ggml_lock_lock       os_unfair_lock_lock
//#define ggml_lock_unlock     os_unfair_lock_unlock
//
//#define GGML_LOCK_INITIALIZER OS_UNFAIR_LOCK_INIT

typedef int ggml_lock_t;

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#define ggml_lock_lock(x)    UNUSED(x)
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0

typedef pthread_t ggml_thread_t;

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

//typedef pthread_spinlock_t ggml_lock_t;

//#define ggml_lock_init(x) pthread_spin_init(x, PTHREAD_PROCESS_PRIVATE)
//#define ggml_lock_destroy pthread_spin_destroy
//#define ggml_lock_lock    pthread_spin_lock
//#define ggml_lock_unlock  pthread_spin_unlock

typedef int ggml_lock_t;

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#define ggml_lock_lock(x)    UNUSED(x)
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0

typedef pthread_t ggml_thread_t;

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

struct ggml_compute_state_shared {
    ggml_lock_t spin;

    int n_threads;

    // synchronization primitives
    atomic_int  n_ready;
    atomic_bool has_work;
    atomic_bool stop; // stop all threads
};

struct ggml_compute_state {
    ggml_thread_t thrd;

    struct ggml_compute_params params;
    struct ggml_tensor * node;

    struct ggml_compute_state_shared * shared;
};

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;

    const int n_threads = state->shared->n_threads;

    while (true) {
        if (atomic_fetch_add(&state->shared->n_ready, 1) == n_threads - 1) {
            atomic_store(&state->shared->has_work, false);
        } else {
            while (atomic_load(&state->shared->has_work)) {
                if (atomic_load(&state->shared->stop)) {
                    return 0;
                }
                ggml_lock_lock  (&state->shared->spin);
                ggml_lock_unlock(&state->shared->spin);
            }
        }

        atomic_fetch_sub(&state->shared->n_ready, 1);

        // wait for work
        while (!atomic_load(&state->shared->has_work)) {
            if (atomic_load(&state->shared->stop)) {
                return 0;
            }
            ggml_lock_lock  (&state->shared->spin);
            ggml_lock_unlock(&state->shared->spin);
        }

        // check if we should stop
        if (atomic_load(&state->shared->stop)) {
            break;
        }

        if (state->node) {
            if (state->params.ith < state->params.nth) {
                ggml_compute_forward(&state->params, state->node);
            }

            state->node = NULL;
        } else {
            break;
        }
    }

    return 0;
}

void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph) {
    const int n_threads = cgraph->n_threads;

    struct ggml_compute_state_shared state_shared = {
        /*.spin      =*/ GGML_LOCK_INITIALIZER,
        /*.n_threads =*/ n_threads,
        /*.n_ready   =*/ 0,
        /*.has_work  =*/ false,
        /*.stop      =*/ false,
    };
    struct ggml_compute_state * workers = n_threads > 1 ? alloca(sizeof(struct ggml_compute_state)*(n_threads - 1)) : NULL;

    // create thread pool
    if (n_threads > 1) {
        ggml_lock_init(&state_shared.spin);

        atomic_store(&state_shared.has_work, true);

        for (int j = 0; j < n_threads - 1; j++) {
            workers[j] = (struct ggml_compute_state) {
                .thrd   = 0,
                .params = {
                    .type  = GGML_TASK_COMPUTE,
                    .ith   = j + 1,
                    .nth   = n_threads,
                    .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                },
                .node   = NULL,
                .shared = &state_shared,
            };

            int rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
            GGML_ASSERT(rc == 0);
            UNUSED(rc);
        }
    }

    // initialize tasks + work buffer
    {
        size_t work_size = 0;

        // thread scheduling for the different operations
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];

            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        node->n_tasks = n_threads;

                        size_t cur = 0;
                        if (ggml_is_quantized(node->type)) {
                            cur = GGML_TYPE_SIZE[GGML_TYPE_F32] * node->ne[0] * n_threads;
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case GGML_OP_ADD:
                    {
                        node->n_tasks = n_threads;

                        size_t cur = 0;

                        if (ggml_is_quantized(node->src0->type)) {
                            cur = GGML_TYPE_SIZE[GGML_TYPE_F32] * node->src0->ne[0] * n_threads;
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case GGML_OP_SUB:
                case GGML_OP_MUL:
                case GGML_OP_DIV:
                case GGML_OP_SQR:
                case GGML_OP_SQRT:
                case GGML_OP_SUM:
                case GGML_OP_MEAN:
                case GGML_OP_REPEAT:
                case GGML_OP_ABS:
                case GGML_OP_SGN:
                case GGML_OP_NEG:
                case GGML_OP_STEP:
                case GGML_OP_RELU:
                    {
                        node->n_tasks = 1;
                    } break;
                case GGML_OP_GELU:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case GGML_OP_SILU:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case GGML_OP_NORM:
                case GGML_OP_RMS_NORM:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        node->n_tasks = n_threads;

                        // TODO: use different scheduling for different matrix sizes
                        //const int nr0 = ggml_nrows(node->src0);
                        //const int nr1 = ggml_nrows(node->src1);

                        //node->n_tasks = MIN(n_threads, MAX(1, nr0/128));
                        //printf("nr0 = %8d, nr1 = %8d, nr0*nr1 = %8d, n_tasks = %d\n", nr0, nr1, nr0*nr1, node->n_tasks);

                        size_t cur = 0;

                        if (node->src0->type == GGML_TYPE_F16 && node->src1->type == GGML_TYPE_F32) {
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CUBLAS)
                            if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                node->n_tasks = 1; // TODO: this actually is doing nothing
                                                   //       the threads are still spinning
                                cur = GGML_TYPE_SIZE[GGML_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                                //printf("src0: ne0 = %d, ne1 = %d, ne = %d\n", node->src0->ne[0], node->src0->ne[1], node->src0->ne[0]*node->src0->ne[1]);
                                //printf("src1: ne0 = %d, ne1 = %d, ne = %d\n", node->src1->ne[0], node->src1->ne[1], node->src1->ne[0]*node->src1->ne[1]);
                                //printf("cur = %zu\n", cur);
                            } else {
                                cur = GGML_TYPE_SIZE[GGML_TYPE_F16]*ggml_nelements(node->src1);
                            }
#else
                            cur = GGML_TYPE_SIZE[GGML_TYPE_F16]*ggml_nelements(node->src1);
#endif
                        } else if (node->src0->type == GGML_TYPE_F32 && node->src1->type == GGML_TYPE_F32) {
                            cur = 0;
                        } else if (ggml_is_quantized(node->src0->type) && node->src1->type == GGML_TYPE_F32) {
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CUBLAS)
                            if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                node->n_tasks = 1;
                                cur = GGML_TYPE_SIZE[GGML_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                            } else
#endif
                            {
                                cur = GGML_TYPE_SIZE[GGML_TYPE_Q8_0]*ggml_nelements(node->src1)/GGML_BLCK_SIZE[GGML_TYPE_Q8_0];
                            }
                        } else {
                            GGML_ASSERT(false);
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case GGML_OP_SCALE:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case GGML_OP_CONT:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                case GGML_OP_GET_ROWS:
                case GGML_OP_DIAG_MASK_INF:
                    {
                        node->n_tasks = 1;
                    } break;
                case GGML_OP_SOFT_MAX:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case GGML_OP_ROPE:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case GGML_OP_CONV_1D_1S:
                case GGML_OP_CONV_1D_2S:
                    {
                        node->n_tasks = n_threads;

                        GGML_ASSERT(node->src0->ne[3] == 1);
                        GGML_ASSERT(node->src1->ne[2] == 1);
                        GGML_ASSERT(node->src1->ne[3] == 1);

                        size_t cur = 0;
                        const int nk = node->src0->ne[0];

                        if (node->src0->type == GGML_TYPE_F16 &&
                            node->src1->type == GGML_TYPE_F32) {
                            cur = sizeof(ggml_fp16_t)*(
                                    nk*ggml_up32(node->src0->ne[1])*node->src0->ne[2] +
                                    ( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
                                    );
                        } else if (node->src0->type == GGML_TYPE_F32 &&
                                   node->src1->type == GGML_TYPE_F32) {
                            cur = sizeof(float)*(
                                    nk*ggml_up32(node->src0->ne[1])*node->src0->ne[2] +
                                    ( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
                                    );
                        } else {
                            GGML_ASSERT(false);
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case GGML_OP_FLASH_ATTN:
                    {
                        node->n_tasks = n_threads;

                        size_t cur = 0;

                        const int64_t ne11 = ggml_up(node->src1->ne[1], GGML_SOFT_MAX_UNROLL);

                        if (node->src1->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
                        }

                        if (node->src1->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case GGML_OP_FLASH_FF:
                    {
                        node->n_tasks = n_threads;

                        size_t cur = 0;

                        if (node->src1->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
                        }

                        if (node->src1->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case GGML_OP_MAP_UNARY:
                case GGML_OP_MAP_BINARY:
                    {
                        node->n_tasks = 1;
                    } break;
                case GGML_OP_NONE:
                    {
                        node->n_tasks = 1;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ASSERT(false);
                    } break;
            }
        }

        if (cgraph->work != NULL && work_size > cgraph->work_size) {
            GGML_ASSERT(false); // TODO: better handling
        }

        if (work_size > 0 && cgraph->work == NULL) {
            cgraph->work_size = work_size + CACHE_LINE_SIZE*(n_threads - 1);

            GGML_PRINT_DEBUG("%s: allocating work buffer for graph (%zu bytes)\n", __func__, cgraph->work_size);
            cgraph->work = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, cgraph->work_size);
        }
    }

    const int64_t perf_start_cycles  = ggml_perf_cycles();
    const int64_t perf_start_time_us = ggml_perf_time_us();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        GGML_PRINT_DEBUG_5("%s: %d/%d\n", __func__, i, cgraph->n_nodes);

        struct ggml_tensor * node = cgraph->nodes[i];

        // TODO: this could be used to avoid unnecessary computations, but it needs to be improved
        //if (node->grad == NULL && node->perf_runs > 0) {
        //    continue;
        //}

        const int64_t perf_node_start_cycles  = ggml_perf_cycles();
        const int64_t perf_node_start_time_us = ggml_perf_time_us();

        // INIT
        struct ggml_compute_params params = {
            /*.type  =*/ GGML_TASK_INIT,
            /*.ith   =*/ 0,
            /*.nth   =*/ node->n_tasks,
            /*.wsize =*/ cgraph->work ? ggml_nbytes(cgraph->work) : 0,
            /*.wdata =*/ cgraph->work ? cgraph->work->data : NULL,
        };

        ggml_compute_forward(&params, node);

        // COMPUTE
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            // launch thread pool
            for (int j = 0; j < n_threads - 1; j++) {
                workers[j].params = (struct ggml_compute_params) {
                    .type  = GGML_TASK_COMPUTE,
                    .ith   = j + 1,
                    .nth   = node->n_tasks,
                    .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                };
                workers[j].node = node;
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) > 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_store(&state_shared.has_work, true);
        }

        params.type = GGML_TASK_COMPUTE;
        ggml_compute_forward(&params, node);

        // wait for thread pool
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) != 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }
        }

        // FINALIZE
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            // launch thread pool
            for (int j = 0; j < n_threads - 1; j++) {
                workers[j].params = (struct ggml_compute_params) {
                    .type  = GGML_TASK_FINALIZE,
                    .ith   = j + 1,
                    .nth   = node->n_tasks,
                    .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                };
                workers[j].node = node;
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) > 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_store(&state_shared.has_work, true);
        }

        params.type = GGML_TASK_FINALIZE;
        ggml_compute_forward(&params, node);

        // wait for thread pool
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) != 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }
        }

        // performance stats (node)
        {
            int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_node_start_cycles;
            int64_t perf_time_us_cur = ggml_perf_time_us() - perf_node_start_time_us;

            node->perf_runs++;
            node->perf_cycles  += perf_cycles_cur;
            node->perf_time_us += perf_time_us_cur;
        }
    }

    // join thread pool
    if (n_threads > 1) {
        atomic_store(&state_shared.stop, true);
        atomic_store(&state_shared.has_work, true);

        for (int j = 0; j < n_threads - 1; j++) {
            int rc = ggml_thread_join(workers[j].thrd, NULL);
            GGML_ASSERT(rc == 0);
            UNUSED(rc);
        }

        ggml_lock_destroy(&state_shared.spin);
    }

    // performance stats (graph)
    {
        int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_start_cycles;
        int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

        cgraph->perf_runs++;
        cgraph->perf_cycles  += perf_cycles_cur;
        cgraph->perf_time_us += perf_time_us_cur;

        GGML_PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
                __func__, cgraph->perf_runs,
                (double) perf_cycles_cur      / (double) ggml_cycles_per_ms(),
                (double) cgraph->perf_cycles  / (double) ggml_cycles_per_ms() / (double) cgraph->perf_runs,
                (double) perf_time_us_cur     / 1000.0,
                (double) cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
    }
}

void ggml_graph_reset(struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * grad = cgraph->grads[i];

        if (grad) {
            ggml_set_zero(grad);
        }
    }
}

void ggml_graph_print(const struct ggml_cgraph * cgraph) {
    int64_t perf_total_per_op_us[GGML_OP_COUNT] = {0};

    GGML_PRINT("=== GRAPH ===\n");

    GGML_PRINT_DEBUG("n_threads       = %d\n",        cgraph->n_threads);
    GGML_PRINT_DEBUG("total work size = %zu bytes\n", cgraph->work_size);

    GGML_PRINT("n_nodes = %d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        perf_total_per_op_us[node->op] += node->perf_time_us;

        GGML_PRINT(" - %3d: [ %" PRId64 ", %" PRId64 ", %" PRId64 "] %16s %s (%3d) cpu = %7.3f / %7.3f ms, wall = %7.3f / %7.3f ms\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                GGML_OP_LABEL[node->op], node->is_param ? "x" : node->grad ? "g" : " ", node->perf_runs,
                (double) node->perf_cycles  / (double) ggml_cycles_per_ms(),
                (double) node->perf_cycles  / (double) ggml_cycles_per_ms() / (double) node->perf_runs,
                (double) node->perf_time_us / 1000.0,
                (double) node->perf_time_us / 1000.0 / node->perf_runs);
    }

    GGML_PRINT("n_leafs = %d\n", cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * node = cgraph->leafs[i];

        GGML_PRINT(" - %3d: [ %" PRId64 ", %" PRId64 "] %8s\n",
                i,
                node->ne[0], node->ne[1],
                GGML_OP_LABEL[node->op]);
    }

    for (int i = 0; i < GGML_OP_COUNT; i++) {
        GGML_PRINT("perf_total_per_op_us[%16s] = %7.3f ms\n", GGML_OP_LABEL[i], (double) perf_total_per_op_us[i] / 1000.0);
    }

    GGML_PRINT("========================================\n");
}

// check if node is part of the graph
static bool ggml_graph_find(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node) {
    if (cgraph == NULL) {
        return true;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return true;
        }
    }

    return false;
}

static struct ggml_tensor * ggml_graph_get_parent(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * parent = cgraph->nodes[i];

        if (parent->grad == node) {
            return parent;
        }
    }

    return NULL;
}

void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename) {
    char color[16];

    FILE * fp = fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = LR;\n");

    for (int i = 0; i < gb->n_nodes; i++) {
        struct ggml_tensor * node = gb->nodes[i];

        if (ggml_graph_get_parent(gb, node) != NULL) {
            continue;
        }

        if (node->is_param) {
            snprintf(color, sizeof(color), "yellow");
        } else if (node->grad) {
            if (ggml_graph_find(gf, node)) {
                snprintf(color, sizeof(color), "green");
            } else {
                snprintf(color, sizeof(color), "lightblue");
            }
        } else {
            snprintf(color, sizeof(color), "white");
        }

        fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"%d [%" PRId64 ", %" PRId64 "] | <x>%s",
                (void *) node, color,
                i, node->ne[0], node->ne[1],
                GGML_OP_SYMBOL[node->op]);

        if (node->grad) {
            fprintf(fp, " | <g>%s\"; ]\n", GGML_OP_SYMBOL[node->grad->op]);
        } else {
            fprintf(fp, "\"; ]\n");
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct ggml_tensor * node = gb->leafs[i];

        snprintf(color, sizeof(color), "pink");

        if (ggml_nelements(node) == 1) {
            fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"<x>%.1e\"; ]\n",
                    (void *) node, color, (double)ggml_get_f32_1d(node, 0));
        } else {
            fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"<x>CONST %d [%" PRId64 ", %" PRId64 "]\"; ]\n",
                    (void *) node, color,
                    i, node->ne[0], node->ne[1]);
        }
    }

    for (int i = 0; i < gb->n_nodes; i++) {
        struct ggml_tensor * node = gb->nodes[i];

        struct ggml_tensor * parent = ggml_graph_get_parent(gb, node);

        if (node->src0) {
            struct ggml_tensor * parent0 = ggml_graph_get_parent(gb, node->src0);

            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"x\"; ]\n",
                    parent0 ? (void *) parent0 : (void *) node->src0,
                    parent0 ? "g" : "x",
                    parent ? (void *) parent : (void *) node,
                    parent ? "g" : "x",
                    parent ? "empty" : "vee",
                    parent ? "dashed" : "solid");
        }

        if (node->src1) {
            struct ggml_tensor * parent1 = ggml_graph_get_parent(gb, node->src1);

            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"y\"; ]\n",
                    parent1 ? (void *) parent1 : (void *) node->src1,
                    parent1 ? "g" : "x",
                    parent ? (void *) parent : (void *) node,
                    parent ? "g" : "x",
                    parent ? "empty" : "vee",
                    parent ? "dashed" : "solid");
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct ggml_tensor * node = gb->leafs[i];

        if (node->src0) {
            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"x\"; ]\n",
                    (void *) node->src0, "x",
                    (void *) node, "x");
        }

        if (node->src1) {
            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"y\"; ]\n",
                    (void *) node->src1, "x",
                    (void *) node, "x");
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    GGML_PRINT("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

static void ggml_opt_set_params(int np, struct ggml_tensor * const ps[], const float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = ggml_nelements(ps[p]) ;
        // TODO: add function to set tensor from array
        for (int64_t j = 0; j < ne; ++j) {
            ggml_set_f32_1d(ps[p], j, x[i++]);
        }
    }
}

static void ggml_opt_get_params(int np, struct ggml_tensor * const ps[], float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            x[i++] = ggml_get_f32_1d(ps[p], j);
        }
    }
}

static void ggml_opt_get_grad(int np, struct ggml_tensor * const ps[], float * g) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            g[i++] = ggml_get_f32_1d(ps[p]->grad, j);
        }
    }
}

//
// ADAM
//
//   ref: https://arxiv.org/pdf/1412.6980.pdf
//

static enum ggml_opt_result ggml_opt_adam(
        struct ggml_context * ctx,
        struct ggml_opt_params params,
        struct ggml_tensor * f,
        struct ggml_cgraph * gf,
        struct ggml_cgraph * gb) {
    GGML_ASSERT(ggml_is_scalar(f));

    gf->n_threads = params.n_threads;
    gb->n_threads = params.n_threads;

    // these will store the parameters we want to optimize
    struct ggml_tensor * ps[GGML_MAX_PARAMS];

    int np = 0;
    int nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->is_param) {
            GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            GGML_ASSERT(np < GGML_MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += ggml_nelements(gf->nodes[i]);
        }
    }

    // constants
    const float alpha = params.adam.alpha;
    const float beta1 = params.adam.beta1;
    const float beta2 = params.adam.beta2;
    const float eps   = params.adam.eps;

    float * x  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // view of the parameters
    float * g1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // gradient
    float * g2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // gradient squared
    float * m  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // first moment
    float * v  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // second moment
    float * mh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // first moment hat
    float * vh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // second moment hat

    float * pf = params.past > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params.past)->data : NULL; // past function values

    // initialize
    ggml_vec_set_f32(nx, m, 0.0f);
    ggml_vec_set_f32(nx, v, 0.0f);

    // update view
    ggml_opt_get_params(np, ps, x);

    // compute the function value
    ggml_graph_reset  (gf);
    ggml_set_f32      (f->grad, 1.0f);
    ggml_graph_compute(ctx, gb);

    float fx_prev = ggml_get_f32_1d(f, 0);
    if (pf) {
        pf[0] = fx_prev;
    }

    int n_no_improvement = 0;
    float fx_best = fx_prev;

    // run the optimizer
    for (int t = 0; t < params.adam.n_iter; ++t) {
        GGML_PRINT_DEBUG  ("=== iter %d ===\n", t);

        GGML_PRINT_DEBUG  ("f      = %10.6f\n", ggml_get_f32_1d(f, 0));
        GGML_PRINT_DEBUG_5("df/dx0 = %10.6f\n", ggml_get_f32_1d(ps[0]->grad, 0));
        GGML_PRINT_DEBUG_5("df/dx1 = %10.6f\n", ggml_get_f32_1d(ps[1]->grad, 0));

        for (int i = 0; i < np; ++i) {
            GGML_PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i,
                    ggml_get_f32_1d(ps[i], 0), ggml_get_f32_1d(ps[i]->grad, 0));
        }

        const int64_t t_start_wall = ggml_time_us();
        const int64_t t_start_cpu = ggml_cycles();
        UNUSED(t_start_wall);
        UNUSED(t_start_cpu);

        {
            // update the gradient
            ggml_opt_get_grad(np, ps, g1);

            // m_t = beta1*m_t-1 + (1 - beta1)*g_t
            ggml_vec_scale_f32(nx, m, beta1);
            ggml_vec_mad_f32  (nx, m, g1, 1.0f - beta1);

            // g2 = g1^2
            ggml_vec_sqr_f32  (nx, g2, g1);

            // v_t = beta2*v_t-1 + (1 - beta2)*g_t^2
            ggml_vec_scale_f32(nx, v, beta2);
            ggml_vec_mad_f32  (nx, v, g2, 1.0f - beta2);

            // m^hat = m_t / (1 - beta1^t)
            // v^hat = v_t / (1 - beta2^t)
            // x_t = x_t-1 - alpha*m^hat/(sqrt(v^hat) + eps)
            ggml_vec_cpy_f32  (nx, mh, m);
            ggml_vec_cpy_f32  (nx, vh, v);

            ggml_vec_scale_f32(nx, mh, alpha/(1.0f - powf(beta1, t + 1)));
            ggml_vec_scale_f32(nx, vh,  1.0f/(1.0f - powf(beta2, t + 1)));

            ggml_vec_sqrt_f32 (nx, vh, vh);
            ggml_vec_acc1_f32 (nx, vh, eps);

            ggml_vec_div_f32  (nx, mh, mh, vh);
            ggml_vec_sub_f32  (nx, x,  x,  mh);

            // update the parameters
            ggml_opt_set_params(np, ps, x);
        }

        ggml_graph_reset  (gf);
        ggml_set_f32      (f->grad, 1.0f);
        ggml_graph_compute(ctx, gb);

        const float fx = ggml_get_f32_1d(f, 0);

        // check convergence
        if (fabsf(fx - fx_prev)/fx < params.adam.eps_f) {
            GGML_PRINT_DEBUG("converged\n");

            return GGML_OPT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= t) {
                const float rate = (pf[t%params.past] - fx)/fx;

                if (fabsf(rate) < params.delta) {
                    return GGML_OPT_OK;
                }
            }

            pf[t%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx_best > fx) {
                fx_best = fx;
                n_no_improvement = 0;
            } else {
                ++n_no_improvement;

                if (n_no_improvement >= params.max_no_improvement) {
                    return GGML_OPT_OK;
                }
            }
        }

        fx_prev = fx;

        {
            const int64_t t_end_cpu = ggml_cycles();
            GGML_PRINT_DEBUG("time iter:      %5.3f s\n", ((float)(t_end_cpu - t_start_cpu))/CLOCKS_PER_SEC);
            UNUSED(t_end_cpu);

            const int64_t t_end_wall = ggml_time_us();
            GGML_PRINT_DEBUG("wall time iter: %5.3f s\n", (t_end_wall - t_start_wall)/1e6);
            UNUSED(t_end_wall);
        }
    }

    return GGML_OPT_DID_NOT_CONVERGE;
}

//
// L-BFGS
//
// the L-BFGS implementation below is based on the following implementation:
//
//   https://github.com/chokkan/liblbfgs
//

struct ggml_lbfgs_iteration_data {
    float alpha;
    float ys;
    float * s;
    float * y;
};

static enum ggml_opt_result linesearch_backtracking(
        struct ggml_context * ctx,
        const struct ggml_opt_params * params,
        int nx,
        float * x,
        float * fx,
        float * g,
        float * d,
        float * step,
        const float * xp,
        struct ggml_tensor * f,
        struct ggml_cgraph * gf,
        struct ggml_cgraph * gb,
        const int np,
        struct ggml_tensor * ps[]) {
    int count = 0;

    float width  = 0.0f;
    float dg     = 0.0f;
    float finit  = 0.0f;
    float dginit = 0.0f;
    float dgtest = 0.0f;

    const float dec = 0.5f;
    const float inc = 2.1f;

    if (*step <= 0.f) {
        return GGML_LINESEARCH_INVALID_PARAMETERS;
    }

    // compute the initial gradient in the search direction
    ggml_vec_dot_f32(nx, &dginit, g, d);

    // make sure that d points to a descent direction
    if (0 < dginit) {
        return GGML_LINESEARCH_FAIL;
    }

    // initialize local variables
    finit = *fx;
    dgtest = params->lbfgs.ftol*dginit;

    while (true) {
        ggml_vec_cpy_f32(nx, x, xp);
        ggml_vec_mad_f32(nx, x, d, *step);

        // evaluate the function and gradient values
        {
            ggml_opt_set_params(np, ps, x);

            ggml_graph_reset  (gf);
            ggml_set_f32      (f->grad, 1.0f);
            ggml_graph_compute(ctx, gb);

            ggml_opt_get_grad(np, ps, g);

            *fx = ggml_get_f32_1d(f, 0);
        }

        ++count;

        if (*fx > finit + (*step)*dgtest) {
            width = dec;
        } else {
            // Armijo condition is satisfied
            if (params->lbfgs.linesearch == GGML_LINESEARCH_BACKTRACKING_ARMIJO) {
                return count;
            }

            ggml_vec_dot_f32(nx, &dg, g, d);

            // check the Wolfe condition
            if (dg < params->lbfgs.wolfe * dginit) {
                width = inc;
            } else {
                if(params->lbfgs.linesearch == GGML_LINESEARCH_BACKTRACKING_WOLFE) {
                    // regular Wolfe conditions
                    return count;
                }

                if(dg > -params->lbfgs.wolfe*dginit) {
                    width = dec;
                } else {
                    // strong Wolfe condition (GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                    return count;
                }
                return count;
            }
        }

        if (*step < params->lbfgs.min_step) {
            return GGML_LINESEARCH_MINIMUM_STEP;
        }
        if (*step > params->lbfgs.max_step) {
            return GGML_LINESEARCH_MAXIMUM_STEP;
        }
        if (params->lbfgs.max_linesearch <= count) {
            return GGML_LINESEARCH_MAXIMUM_ITERATIONS;
        }

        (*step) *= width;
    }

    return GGML_LINESEARCH_FAIL;
}

static enum ggml_opt_result ggml_opt_lbfgs(
        struct ggml_context * ctx,
        struct ggml_opt_params params,
        struct ggml_tensor * f,
        struct ggml_cgraph * gf,
        struct ggml_cgraph * gb) {
    if (params.lbfgs.linesearch == GGML_LINESEARCH_BACKTRACKING_WOLFE ||
        params.lbfgs.linesearch == GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
        if (params.lbfgs.wolfe <= params.lbfgs.ftol || 1.f <= params.lbfgs.wolfe) {
            return GGML_OPT_INVALID_WOLFE;
        }
    }

    gf->n_threads = params.n_threads;
    gb->n_threads = params.n_threads;

    const int m = params.lbfgs.m;

    // these will store the parameters we want to optimize
    struct ggml_tensor * ps[GGML_MAX_PARAMS];

    int np = 0;
    int nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->is_param) {
            GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            GGML_ASSERT(np < GGML_MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += ggml_nelements(gf->nodes[i]);
        }
    }

    float * x  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // current parameters
    float * xp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // previous parameters
    float * g  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // current gradient
    float * gp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // previous gradient
    float * d  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data; // search direction

    float * pf = params.past > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params.past)->data : NULL; // past function values

    float fx    = 0.0f; // cost function value
    float xnorm = 0.0f; // ||x||
    float gnorm = 0.0f; // ||g||
    float step  = 0.0f;

    // initialize x from the graph nodes
    ggml_opt_get_params(np, ps, x);

    // the L-BFGS memory
    struct ggml_lbfgs_iteration_data * lm = alloca(sizeof(struct ggml_lbfgs_iteration_data)*m);

    for (int i = 0; i < m; ++i) {
        lm[i].alpha = 0.0f;
        lm[i].ys    = 0.0f;
        lm[i].s     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data;
        lm[i].y     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nx)->data;
    }

    // evaluate the function value and its gradient
    {
        ggml_opt_set_params(np, ps, x);

        ggml_graph_reset  (gf);
        ggml_set_f32      (f->grad, 1.0f);
        ggml_graph_compute(ctx, gb);

        ggml_opt_get_grad(np, ps, g);

        fx = ggml_get_f32_1d(f, 0);
    }

    if (pf) {
        pf[0] = fx;
    }

    float fx_best = fx;

    // search direction = -gradient
    ggml_vec_neg_f32(nx, d, g);

    // ||x||, ||g||
    ggml_vec_norm_f32(nx, &xnorm, x);
    ggml_vec_norm_f32(nx, &gnorm, g);

    if (xnorm < 1.0f) {
        xnorm = 1.0f;
    }

    // already optimized
    if (gnorm/xnorm <= params.lbfgs.eps) {
        return GGML_OPT_OK;
    }

    // initial step
    ggml_vec_norm_inv_f32(nx, &step, d);

    int j                = 0;
    int k                = 1;
    int ls               = 0;
    int end              = 0;
    int bound            = 0;
    int n_no_improvement = 0;

    float ys   = 0.0f;
    float yy   = 0.0f;
    float beta = 0.0f;

    while (true) {
        // store the current position and gradient vectors
        ggml_vec_cpy_f32(nx, xp, x);
        ggml_vec_cpy_f32(nx, gp, g);

        ls = linesearch_backtracking(ctx, &params, nx, x, &fx, g, d, &step, xp, f, gf, gb, np, ps);

        if (ls < 0) {
            // linesearch failed - go back to the previous point and return
            ggml_vec_cpy_f32(nx, x, xp);
            ggml_vec_cpy_f32(nx, g, gp);

            return ls;
        }

        ggml_vec_norm_f32(nx, &xnorm, x);
        ggml_vec_norm_f32(nx, &gnorm, g);

        GGML_PRINT_DEBUG("f = %10.6f\n", ggml_get_f32_1d(f, 0));

        if (xnorm < 1.0f) {
            xnorm = 1.0f;
        }
        if (gnorm/xnorm <= params.lbfgs.eps) {
            // converged
            return GGML_OPT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= k) {
                const float rate = (pf[k%params.past] - fx)/fx;

                if (fabsf(rate) < params.delta) {
                    return GGML_OPT_OK;
                }
            }

            pf[k%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx < fx_best) {
                fx_best = fx;
                n_no_improvement = 0;
            } else {
                n_no_improvement++;

                if (n_no_improvement >= params.max_no_improvement) {
                    return GGML_OPT_OK;
                }
            }
        }

        if (params.lbfgs.n_iter != 0 && params.lbfgs.n_iter < k + 1) {
            // reached the maximum number of iterations
            return GGML_OPT_DID_NOT_CONVERGE;
        }

        // update vectors s and y:
        //   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
        //   y_{k+1} = g_{k+1} - g_{k}.
        //
        ggml_vec_sub_f32(nx, lm[end].s, x, xp);
        ggml_vec_sub_f32(nx, lm[end].y, g, gp);

        // compute scalars ys and yy:
        //     ys = y^t \cdot s    -> 1 / \rho.
        //     yy = y^t \cdot y.
        //
        ggml_vec_dot_f32(nx, &ys, lm[end].y, lm[end].s);
        ggml_vec_dot_f32(nx, &yy, lm[end].y, lm[end].y);

        lm[end].ys = ys;

        // find new search direction
        //   ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS

        bound = (m <= k) ? m : k;
        k++;
        end = (end + 1)%m;

        // initialize search direction with -g
        ggml_vec_neg_f32(nx, d, g);

        j = end;
        for (int i = 0; i < bound; ++i) {
            j = (j + m - 1) % m;
            // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
            ggml_vec_dot_f32(nx, &lm[j].alpha, lm[j].s, d);
            lm[j].alpha /= lm[j].ys;
            // q_{i} = q_{i+1} - \alpha_{i} y_{i}
            ggml_vec_mad_f32(nx, d, lm[j].y, -lm[j].alpha);
        }

        ggml_vec_scale_f32(nx, d, ys/yy);

        for (int i = 0; i < bound; ++i) {
            // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
            ggml_vec_dot_f32(nx, &beta, lm[j].y, d);
            beta /= lm[j].ys;
            // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
            ggml_vec_mad_f32(nx, d, lm[j].s, lm[j].alpha - beta);
            j = (j + 1)%m;
        }

        step = 1.0;
    }

    return GGML_OPT_DID_NOT_CONVERGE;
}

struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type) {
    struct ggml_opt_params result;

    switch (type) {
        case GGML_OPT_ADAM:
            {
                result = (struct ggml_opt_params) {
                    .type      = GGML_OPT_ADAM,
                    .n_threads = 1,
                    .past      = 0,
                    .delta     = 1e-5f,

                    .max_no_improvement = 100,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .adam = {
                        .n_iter = 10000,
                        .alpha  = 0.001f,
                        .beta1  = 0.9f,
                        .beta2  = 0.999f,
                        .eps    = 1e-8f,
                        .eps_f  = 1e-5f,
                        .eps_g  = 1e-3f,
                    },
                };
            } break;
        case GGML_OPT_LBFGS:
            {
                result = (struct ggml_opt_params) {
                    .type      = GGML_OPT_LBFGS,
                    .n_threads = 1,
                    .past      = 0,
                    .delta     = 1e-5f,

                    .max_no_improvement = 0,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .lbfgs = {
                        .m              = 6,
                        .n_iter         = 100,
                        .max_linesearch = 20,

                        .eps      = 1e-5f,
                        .ftol     = 1e-4f,
                        .wolfe    = 0.9f,
                        .min_step = 1e-20f,
                        .max_step = 1e+20f,

                        .linesearch = GGML_LINESEARCH_DEFAULT,
                    },
                };
            } break;
    }

    return result;
}

enum ggml_opt_result ggml_opt(
        struct ggml_context * ctx,
        struct ggml_opt_params params,
        struct ggml_tensor * f) {
    bool free_ctx = false;
    if (ctx == NULL) {
        struct ggml_init_params params_ctx = {
            .mem_size   = 16*1024*1024,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };

        ctx = ggml_init(params_ctx);
        if (ctx == NULL) {
            return GGML_OPT_NO_CONTEXT;
        }

        free_ctx = true;
    }

    enum ggml_opt_result result = GGML_OPT_OK;

    // build forward + backward compute graphs
    struct ggml_cgraph gf = ggml_build_forward (f);
    struct ggml_cgraph gb = ggml_build_backward(ctx, &gf, false);

    switch (params.type) {
        case GGML_OPT_ADAM:
            {
                result = ggml_opt_adam(ctx, params, f, &gf, &gb);
            } break;
        case GGML_OPT_LBFGS:
            {
                result = ggml_opt_lbfgs(ctx, params, f, &gf, &gb);
            } break;
    }

    if (params.print_forward_graph) {
        ggml_graph_print   (&gf);
        ggml_graph_dump_dot(&gf, NULL, "opt-forward.dot");
    }

    if (params.print_backward_graph) {
        ggml_graph_print   (&gb);
        ggml_graph_dump_dot(&gb, &gf, "opt-backward.dot");
    }

    if (free_ctx) {
        ggml_free(ctx);
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////

size_t ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    for (int j = 0; j < n; j += k) {
        block_q4_0 * restrict y = (block_q4_0 *)dst + j/QK4_0;

        quantize_row_q4_0_reference(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK4_0; l += 2) {
                const uint8_t vi0 = y[i].qs[l/2] & 0xF;
                const uint8_t vi1 = y[i].qs[l/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_0*sizeof(block_q4_0));
}

size_t ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK4_1 == 0);
    const int nb = k / QK4_1;

    for (int j = 0; j < n; j += k) {
        block_q4_1 * restrict y = (block_q4_1 *)dst + j/QK4_1;

        quantize_row_q4_1_reference(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK4_1; l += 2) {
                const uint8_t vi0 = y[i].qs[l/2] & 0xF;
                const uint8_t vi1 = y[i].qs[l/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_1*sizeof(block_q4_1));
}

size_t ggml_quantize_q4_2(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK4_2 == 0);
    const int nb = k / QK4_2;

    for (int j = 0; j < n; j += k) {
        block_q4_2 * restrict y = (block_q4_2 *)dst + j/QK4_2;

        //quantize_row_q4_2_reference(src + j, y, k);
        quantize_row_q4_2_rmse(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK4_2; l += 2) {
                const uint8_t vi0 = y[i].qs[l/2] & 0xF;
                const uint8_t vi1 = y[i].qs[l/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_2*sizeof(block_q4_2));
}

size_t ggml_quantize_q4_3(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK4_3 == 0);
    const int nb = k / QK4_3;

    for (int j = 0; j < n; j += k) {
        block_q4_3 * restrict y = (block_q4_3 *)dst + j/QK4_3;

        quantize_row_q4_3_reference(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK4_3; l += 2) {
                const uint8_t vi0 = y[i].qs[l/2] & 0xF;
                const uint8_t vi1 = y[i].qs[l/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_3*sizeof(block_q4_3));
}

size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist) {
    size_t result = 0;
    switch (type) {
        case GGML_TYPE_Q4_0:
            {
                GGML_ASSERT(start % QK4_0 == 0);
                block_q4_0 * block = (block_q4_0*)dst + start / QK4_0;
                result = ggml_quantize_q4_0(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q4_1:
            {
                GGML_ASSERT(start % QK4_1 == 0);
                block_q4_1 * block = (block_q4_1*)dst + start / QK4_1;
                result = ggml_quantize_q4_1(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q4_2:
            {
                GGML_ASSERT(start % QK4_2 == 0);
                block_q4_2 * block = (block_q4_2*)dst + start / QK4_2;
                result = ggml_quantize_q4_2(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q4_3:
            {
                GGML_ASSERT(start % QK4_3 == 0);
                block_q4_3 * block = (block_q4_3*)dst + start / QK4_3;
                result = ggml_quantize_q4_3(src + start, block, n, n, hist);
            } break;
        default:
            assert(false);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_blas(void) {
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CUBLAS)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_cublas(void) {
#if defined(GGML_USE_CUBLAS)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////
