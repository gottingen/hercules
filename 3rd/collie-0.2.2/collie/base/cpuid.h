// Copyright 2024 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef COLLIE_BASE_CPUID_H_
#define COLLIE_BASE_CPUID_H_

#include <algorithm>
#include <cstring>

#if defined(__linux__) && (defined(__ARM_NEON) || defined(_M_ARM) || defined(__riscv_vector))
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#if defined(_MSC_VER)
// Contains the definition of __cpuidex
#include <intrin.h>
#endif

namespace collie {

    struct Cpuid {
        unsigned sse2: 1;
        unsigned sse3: 1;
        unsigned ssse3: 1;
        unsigned sse4_1: 1;
        unsigned sse4_2: 1;
        unsigned sse4a: 1;
        unsigned fma3_sse: 1;
        unsigned fma4: 1;
        unsigned xop: 1;
        unsigned avx: 1;
        unsigned fma3_avx: 1;
        unsigned avx2: 1;
        unsigned avxvnni: 1;
        unsigned fma3_avx2: 1;
        unsigned avx512f: 1;
        unsigned avx512cd: 1;
        unsigned avx512dq: 1;
        unsigned avx512bw: 1;
        unsigned avx512er: 1;
        unsigned avx512pf: 1;
        unsigned avx512ifma: 1;
        unsigned avx512vbmi: 1;
        unsigned avx512vnni_bw: 1;
        unsigned avx512vnni_vbmi: 1;
        unsigned neon: 1;
        unsigned neon64: 1;
        unsigned sve: 1;
        unsigned rvv: 1;

        inline Cpuid() noexcept {
            memset(this, 0, sizeof(Cpuid));

#if defined(__aarch64__) || defined(_M_ARM64)
            neon = 1;
            neon64 = 1;
#elif defined(__ARM_NEON) || defined(_M_ARM)

#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
            neon = bool(getauxval(AT_HWCAP) & HWCAP_NEON);
#else
            // that's very conservative :-/
            neon = 0;
#endif
            neon64 = 0;

#elif defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0

#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
            sve = bool(getauxval(AT_HWCAP) & HWCAP_SVE);
#else
            sve = 0;
#endif

#elif defined(__riscv_vector) && defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen > 0

#if defined(__linux__) && (!defined(__ANDROID_API__) || __ANDROID_API__ >= 18)
#ifndef HWCAP_V
#define HWCAP_V (1 << ('V' - 'A'))
#endif
            rvv = bool(getauxval(AT_HWCAP) & HWCAP_V);
#else
            rvv = 0;
#endif

#elif defined(__x86_64__) || defined(__i386__) || defined(_M_AMD64) || defined(_M_IX86)
            auto get_cpuid = [](int reg[4], int level, int count = 0) noexcept {

#if defined(_MSC_VER)
                __cpuidex(reg, level, count);

#elif defined(__INTEL_COMPILER)
                __cpuid(reg, level);

#elif defined(__GNUC__) || defined(__clang__)

#if defined(__i386__) && defined(__PIC__)
                // %ebx may be the PIC register
                __asm__("xchg{l}\t{%%}ebx, %1\n\t"
                        "cpuid\n\t"
                        "xchg{l}\t{%%}ebx, %1\n\t"
                        : "=a"(reg[0]), "=r"(reg[1]), "=c"(reg[2]),
                          "=d"(reg[3])
                        : "0"(level), "2"(count));

#else
                __asm__("cpuid\n\t"
                        : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]),
                "=d"(reg[3])
                        : "0"(level), "2"(count));
#endif

#else
#error "Unsupported configuration"
#endif
            };

            int regs1[4];

            get_cpuid(regs1, 0x1);

            sse2 = regs1[3] >> 26 & 1;
            sse3 = regs1[2] >> 0 & 1;
            ssse3 = regs1[2] >> 9 & 1;
            sse4_1 = regs1[2] >> 19 & 1;
            sse4_2 = regs1[2] >> 20 & 1;
            fma3_sse = regs1[2] >> 12 & 1;
            avx = regs1[2] >> 28 & 1;
            fma3_avx = avx && fma3_sse;
            int regs8[4];
            get_cpuid(regs8, 0x80000001);
            fma4 = regs8[2] >> 16 & 1;
            int regs7[4];
            get_cpuid(regs7, 0x7);
            avx2 = regs7[1] >> 5 & 1;

            int regs7a[4];
            get_cpuid(regs7a, 0x7, 0x1);
            avxvnni = regs7a[0] >> 4 & 1;

            fma3_avx2 = avx2 && fma3_sse;
            avx512f = regs7[1] >> 16 & 1;
            avx512cd = regs7[1] >> 28 & 1;
            avx512dq = regs7[1] >> 17 & 1;
            avx512bw = regs7[1] >> 30 & 1;
            avx512er = regs7[1] >> 27 & 1;
            avx512pf = regs7[1] >> 26 & 1;
            avx512ifma = regs7[1] >> 21 & 1;
            avx512vbmi = regs7[2] >> 1 & 1;
            avx512vnni_bw = regs7[2] >> 11 & 1;
            avx512vnni_vbmi = avx512vbmi && avx512vnni_bw;
#endif
        }
    };

}  // namespace collie

#endif  // COLLIE_BASE_CPUID_H_
