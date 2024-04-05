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

#ifndef COLLIE_SIMD_ARCH_ISA_H_
#define COLLIE_SIMD_ARCH_ISA_H_

#include <collie/simd/config/simd_arch.h>

#include <collie/simd/arch/generic_fwd.h>

#if COLLIE_SIMD_WITH_SSE2
#include <collie/simd/arch/sse2.h>
#endif

#if COLLIE_SIMD_WITH_SSE3
#include <collie/simd/arch/sse3.h>
#endif

#if COLLIE_SIMD_WITH_SSSE3
#include <collie/simd/arch/ssse3.h>
#endif

#if COLLIE_SIMD_WITH_SSE4_1
#include <collie/simd/arch/sse4_1.h>
#endif

#if COLLIE_SIMD_WITH_SSE4_2
#include <collie/simd/arch/sse4_2.h>
#endif

#if COLLIE_SIMD_WITH_FMA3_SSE
#include <collie/simd/arch/fma3_sse.h>
#endif

#if COLLIE_SIMD_WITH_FMA4
#include <collie/simd/arch/fma4.h>
#endif

#if COLLIE_SIMD_WITH_AVX
#include <collie/simd/arch/avx.h>
#endif

#if COLLIE_SIMD_WITH_FMA3_AVX
#include <collie/simd/arch/fma3_avx.h>
#endif

#if COLLIE_SIMD_WITH_AVXVNNI
#include <collie/simd/arch/avxvnni.h>
#endif

#if COLLIE_SIMD_WITH_AVX2
#include <collie/simd/arch/avx2.h>
#endif

#if COLLIE_SIMD_WITH_FMA3_AVX2
#include <collie/simd/arch/fma3_avx2.h>
#endif

#if COLLIE_SIMD_AVX512F
#include <collie/simd/arch/avx512f.h>
#endif

#if COLLIE_SIMD_AVX512BW
#include <collie/simd/arch/avx512bw.h>
#endif

#if COLLIE_SIMD_AVX512ER
#include <collie/simd/arch/avx512er.h>
#endif

#if COLLIE_SIMD_AVX512PF
#include <collie/simd/arch/avx512pf.h>
#endif

#if COLLIE_SIMD_AVX512IFMA
#include <collie/simd/arch/avx512ifma.h>
#endif

#if COLLIE_SIMD_AVX512VBMI
#include <collie/simd/arch/avx512vbmi.h>
#endif

#if COLLIE_SIMD_WITH_AVX512VNNI_AVX512BW
#include <collie/simd/arch/avx512vnni_avx512bw.h>
#endif

#if COLLIE_SIMD_WITH_AVX512VNNI_AVX512VBMI
#include <collie/simd/arch/avx512vnni_avx512vbmi.h>
#endif

#if COLLIE_SIMD_WITH_NEON
#include <collie/simd/arch/neon.h>
#endif

#if COLLIE_SIMD_WITH_NEON64
#include <collie/simd/arch/neon64.h>
#endif

#if COLLIE_SIMD_WITH_SVE
#include <collie/simd/arch/sve.h>
#endif

#if COLLIE_SIMD_WITH_RVV
#include <collie/simd/arch/rvv.h>
#endif

#if COLLIE_SIMD_WASM
#include <collie/simd/arch/wasm.h>
#endif

// Must come last to have access to all conversion specializations.
#include <collie/simd/arch/generic.h>

#endif  // COLLIE_SIMD_ARCH_ISA_H_
