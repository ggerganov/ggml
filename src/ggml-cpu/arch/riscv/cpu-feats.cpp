#include "ggml-backend-impl.h"

#if defined(__riscv) && __riscv_xlen == 64
#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>

// Older kernel headers may not define every Zv* probe bit. Provide
// fallbacks so the build works against any kernel >= 6.5.
#ifndef RISCV_HWPROBE_EXT_ZVBB
#define RISCV_HWPROBE_EXT_ZVBB     (1ULL << 17)
#endif
#ifndef RISCV_HWPROBE_EXT_ZVBC
#define RISCV_HWPROBE_EXT_ZVBC     (1ULL << 18)
#endif
#ifndef RISCV_HWPROBE_EXT_ZVKB
#define RISCV_HWPROBE_EXT_ZVKB     (1ULL << 19)
#endif
#ifndef RISCV_HWPROBE_EXT_ZVKG
#define RISCV_HWPROBE_EXT_ZVKG     (1ULL << 20)
#endif
#ifndef RISCV_HWPROBE_EXT_ZVKNED
#define RISCV_HWPROBE_EXT_ZVKNED   (1ULL << 21)
#endif
#ifndef RISCV_HWPROBE_EXT_ZVKNHA
#define RISCV_HWPROBE_EXT_ZVKNHA   (1ULL << 22)
#endif
#ifndef RISCV_HWPROBE_EXT_ZVKNHB
#define RISCV_HWPROBE_EXT_ZVKNHB   (1ULL << 23)
#endif
#ifndef RISCV_HWPROBE_EXT_ZVFH
#define RISCV_HWPROBE_EXT_ZVFH     (1ULL << 30)
#endif
#ifndef RISCV_HWPROBE_EXT_ZVFHMIN
#define RISCV_HWPROBE_EXT_ZVFHMIN  (1ULL << 31)
#endif

struct riscv64_features {
    bool has_rvv      = false;
    bool has_zvbb     = false;
    bool has_zvbc     = false;
    bool has_zvkb     = false;
    bool has_zvkg     = false;
    bool has_zvkned   = false;
    bool has_zvknha   = false;
    bool has_zvknhb   = false;
    bool has_zvfh     = false;
    bool has_zvfhmin  = false;

    riscv64_features() {
        struct riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        probe.value = 0;

        int ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0);

        if (0 == ret) {
            has_rvv     = !!(probe.value & RISCV_HWPROBE_IMA_V);
            has_zvbb    = !!(probe.value & RISCV_HWPROBE_EXT_ZVBB);
            has_zvbc    = !!(probe.value & RISCV_HWPROBE_EXT_ZVBC);
            has_zvkb    = !!(probe.value & RISCV_HWPROBE_EXT_ZVKB);
            has_zvkg    = !!(probe.value & RISCV_HWPROBE_EXT_ZVKG);
            has_zvkned  = !!(probe.value & RISCV_HWPROBE_EXT_ZVKNED);
            has_zvknha  = !!(probe.value & RISCV_HWPROBE_EXT_ZVKNHA);
            has_zvknhb  = !!(probe.value & RISCV_HWPROBE_EXT_ZVKNHB);
            has_zvfh    = !!(probe.value & RISCV_HWPROBE_EXT_ZVFH);
            has_zvfhmin = !!(probe.value & RISCV_HWPROBE_EXT_ZVFHMIN);
        }
    }
};

static int ggml_backend_cpu_riscv64_score() {
    int score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV
    if (!rf.has_rvv) { return 0; }
    score += 1 << 1;
#endif

#ifdef __riscv_zvbb
    if (!rf.has_zvbb)    { return 0; }
    score += 1 << 2;
#endif
#ifdef __riscv_zvbc
    if (!rf.has_zvbc)    { return 0; }
    score += 1 << 3;
#endif
#ifdef __riscv_zvkb
    if (!rf.has_zvkb)    { return 0; }
    score += 1 << 4;
#endif
#ifdef __riscv_zvkg
    if (!rf.has_zvkg)    { return 0; }
    score += 1 << 5;
#endif
#ifdef __riscv_zvkned
    if (!rf.has_zvkned)  { return 0; }
    score += 1 << 6;
#endif
#ifdef __riscv_zvknhb
    if (!rf.has_zvknhb)  { return 0; }
    score += 1 << 7;
#elif defined(__riscv_zvknha)
    if (!rf.has_zvknha)  { return 0; }
    score += 1 << 7;
#endif
#ifdef __riscv_zvfh
    if (!rf.has_zvfh)    { return 0; }
    score += 1 << 8;
#elif defined(__riscv_zvfhmin)
    if (!rf.has_zvfhmin) { return 0; }
    score += 1 << 8;
#endif

    return score;
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64
