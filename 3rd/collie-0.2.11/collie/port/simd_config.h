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

#ifndef COLLIE_PORT_SIMD_CONFIG_H_
#define COLLIE_PORT_SIMD_CONFIG_H_

/**
 * high level free functions
 *
 * @defgroup collie_simd_config_macro Instruction Set Detection
 */

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if SSE2 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE2__
#define COLLIE_SIMD_WITH_SSE2 1
#else
#define COLLIE_SIMD_WITH_SSE2 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if SSE3 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE3__
#define COLLIE_SIMD_WITH_SSE3 1
#else
#define COLLIE_SIMD_WITH_SSE3 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if SSSE3 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSSE3__
#define COLLIE_SIMD_WITH_SSSE3 1
#else
#define COLLIE_SIMD_WITH_SSSE3 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if SSE4.1 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE4_1__
#define COLLIE_SIMD_WITH_SSE4_1 1
#else
#define COLLIE_SIMD_WITH_SSE4_1 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if SSE4.2 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE4_2__
#define COLLIE_SIMD_WITH_SSE4_2 1
#else
#define COLLIE_SIMD_WITH_SSE4_2 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX__
#define COLLIE_SIMD_WITH_AVX 1
#else
#define COLLIE_SIMD_WITH_AVX 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX2 is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX2__
#define COLLIE_SIMD_WITH_AVX2 1
#else
#define COLLIE_SIMD_WITH_AVX2 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVXVNNI is available at compile-time, to 0 otherwise.
 */
#ifdef __AVXVNNI__
#define COLLIE_SIMD_WITH_AVXVNNI 1
#else
#define COLLIE_SIMD_WITH_AVXVNNI 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if FMA3 for SSE is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA__

#if defined(__SSE__)
#ifndef COLLIE_SIMD_WITH_FMA3_SSE // Leave the opportunity to manually disable it, see #643
#define COLLIE_SIMD_WITH_FMA3_SSE 1
#endif
#else

#if COLLIE_SIMD_WITH_FMA3_SSE
#error "Manually set COLLIE_SIMD_WITH_FMA3_SSE is incompatible with current compiler flags"
#endif

#define COLLIE_SIMD_WITH_FMA3_SSE 0
#endif

#else

#if COLLIE_SIMD_WITH_FMA3_SSE
#error "Manually set COLLIE_SIMD_WITH_FMA3_SSE is incompatible with current compiler flags"
#endif

#define COLLIE_SIMD_WITH_FMA3_SSE 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if FMA3 for AVX is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA__

#if defined(__AVX__)
#ifndef COLLIE_SIMD_WITH_FMA3_AVX // Leave the opportunity to manually disable it, see #643
#define COLLIE_SIMD_WITH_FMA3_AVX 1
#endif
#else

#if COLLIE_SIMD_WITH_FMA3_AVX
#error "Manually set COLLIE_SIMD_WITH_FMA3_AVX is incompatible with current compiler flags"
#endif

#define COLLIE_SIMD_WITH_FMA3_AVX 0
#endif

#if defined(__AVX2__)
#ifndef COLLIE_SIMD_WITH_FMA3_AVX2 // Leave the opportunity to manually disable it, see #643
#define COLLIE_SIMD_WITH_FMA3_AVX2 1
#endif
#else

#if COLLIE_SIMD_WITH_FMA3_AVX2
#error "Manually set COLLIE_SIMD_WITH_FMA3_AVX2 is incompatible with current compiler flags"
#endif

#define COLLIE_SIMD_WITH_FMA3_AVX2 0
#endif

#else

#if COLLIE_SIMD_WITH_FMA3_AVX
#error "Manually set COLLIE_SIMD_WITH_FMA3_AVX is incompatible with current compiler flags"
#endif

#if COLLIE_SIMD_WITH_FMA3_AVX2
#error "Manually set COLLIE_SIMD_WITH_FMA3_AVX2 is incompatible with current compiler flags"
#endif

#define COLLIE_SIMD_WITH_FMA3_AVX 0
#define COLLIE_SIMD_WITH_FMA3_AVX2 0

#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if FMA4 is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA4__
#define COLLIE_SIMD_WITH_FMA4 1
#else
#define COLLIE_SIMD_WITH_FMA4 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512F is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512F__
// AVX512 instructions are supported starting with gcc 6
// see https://www.gnu.org/software/gcc/gcc-6/changes.html
// check clang first, newer clang always defines __GNUC__ = 4
#if defined(__clang__) && __clang_major__ >= 6
#define COLLIE_SIMD_AVX512F 1
#elif defined(__GNUC__) && __GNUC__ < 6
#define COLLIE_SIMD_AVX512F 0
#else
#define COLLIE_SIMD_AVX512F 1
#if __GNUC__ == 6
#define COLLIE_SIMD_AVX512_SHIFT_INTRINSICS_IMM_ONLY 1
#endif
#endif
#else
#define COLLIE_SIMD_AVX512F 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512CD is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512CD__
// Avoids repeating the GCC workaround over and over
#define COLLIE_SIMD_AVX512CD COLLIE_SIMD_AVX512F
#else
#define COLLIE_SIMD_AVX512CD 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512DQ is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512DQ__
#define COLLIE_SIMD_AVX512DQ COLLIE_SIMD_AVX512F
#else
#define COLLIE_SIMD_AVX512DQ 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512BW is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512BW__
#define COLLIE_SIMD_AVX512BW COLLIE_SIMD_AVX512F
#else
#define COLLIE_SIMD_AVX512BW 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512ER is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512ER__
#define COLLIE_SIMD_AVX512ER COLLIE_SIMD_AVX512F
#else
#define COLLIE_SIMD_AVX512ER 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512PF is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512PF__
#define COLLIE_SIMD_AVX512PF COLLIE_SIMD_AVX512F
#else
#define COLLIE_SIMD_AVX512PF 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512IFMA is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512IFMA__
#define COLLIE_SIMD_AVX512IFMA COLLIE_SIMD_AVX512F
#else
#define COLLIE_SIMD_AVX512IFMA 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512VBMI is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512VBMI__
#define COLLIE_SIMD_AVX512VBMI COLLIE_SIMD_AVX512F
#else
#define COLLIE_SIMD_AVX512VBMI 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if AVX512VNNI is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512VNNI__

#if COLLIE_SIMD_WITH_AVX512_VBMI
#define COLLIE_SIMD_WITH_AVX512VNNI_AVX512VBMI COLLIE_SIMD_AVX512F
#define COLLIE_SIMD_WITH_AVX512VNNI_AVX512BW COLLIE_SIMD_AVX512F
#else
#define COLLIE_SIMD_WITH_AVX512VNNI_AVX512VBMI 0
#define COLLIE_SIMD_WITH_AVX512VNNI_AVX512BW COLLIE_SIMD_AVX512F
#endif

#else

#define COLLIE_SIMD_WITH_AVX512VNNI_AVX512VBMI 0
#define COLLIE_SIMD_WITH_AVX512VNNI_AVX512BW 0

#endif

#ifdef __ARM_NEON

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if NEON is available at compile-time, to 0 otherwise.
 */
#if __ARM_ARCH >= 7
#define COLLIE_SIMD_WITH_NEON 1
#else
#define COLLIE_SIMD_WITH_NEON 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if NEON64 is available at compile-time, to 0 otherwise.
 */
#ifdef __aarch64__
#define COLLIE_SIMD_WITH_NEON64 1
#else
#define COLLIE_SIMD_WITH_NEON64 0
#endif
#else
#define COLLIE_SIMD_WITH_NEON 0
#define COLLIE_SIMD_WITH_NEON64 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if SVE is available and bit width is pre-set at compile-time, to 0 otherwise.
 */
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
#define COLLIE_SIMD_WITH_SVE 1
#define COLLIE_SIMD_SVE_BITS __ARM_FEATURE_SVE_BITS
#else
#define COLLIE_SIMD_WITH_SVE 0
#define COLLIE_SIMD_SVE_BITS 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if RVV is available and bit width is pre-set at compile-time, to 0 otherwise.
 */
#if defined(__riscv_vector) && defined(__riscv_v_fixed_vlen) && __riscv_v_fixed_vlen > 0
#define COLLIE_SIMD_WITH_RVV 1
#define COLLIE_SIMD_RVV_BITS __riscv_v_fixed_vlen
#else
#define COLLIE_SIMD_WITH_RVV 0
#define COLLIE_SIMD_RVV_BITS 0
#endif

/**
 * @ingroup collie_simd_config_macro
 *
 * Set to 1 if WebAssembly SIMD is available at compile-time, to 0 otherwise.
 */
#ifdef __EMSCRIPTEN__
#define COLLIE_SIMD_WASM 1
#else
#define COLLIE_SIMD_WASM 0
#endif

// Workaround for MSVC compiler
#ifdef _MSC_VER

#if COLLIE_SIMD_WITH_AVX512

#undef COLLIE_SIMD_WITH_AVX2
#define COLLIE_SIMD_WITH_AVX2 1

#endif

#if COLLIE_SIMD_WITH_AVX2

#undef COLLIE_SIMD_WITH_AVX
#define COLLIE_SIMD_WITH_AVX 1

#undef COLLIE_SIMD_WITH_FMA3_AVX
#define COLLIE_SIMD_WITH_FMA3_AVX 1

#undef COLLIE_SIMD_WITH_FMA3_AVX2
#define COLLIE_SIMD_WITH_FMA3_AVX2 1

#endif

#if COLLIE_SIMD_WITH_AVX

#undef COLLIE_SIMD_WITH_SSE4_2
#define COLLIE_SIMD_WITH_SSE4_2 1

#endif

#if COLLIE_SIMD_WITH_SSE4_2

#undef COLLIE_SIMD_WITH_SSE4_1
#define COLLIE_SIMD_WITH_SSE4_1 1

#endif

#if COLLIE_SIMD_WITH_SSE4_1

#undef COLLIE_SIMD_WITH_SSSE3
#define COLLIE_SIMD_WITH_SSSE3 1

#endif

#if COLLIE_SIMD_WITH_SSSE3

#undef COLLIE_SIMD_WITH_SSE3
#define COLLIE_SIMD_WITH_SSE3 1

#endif

#if COLLIE_SIMD_WITH_SSE3 || defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#undef COLLIE_SIMD_WITH_SSE2
#define COLLIE_SIMD_WITH_SSE2 1
#endif

#endif

#if !COLLIE_SIMD_WITH_SSE2 && !COLLIE_SIMD_WITH_SSE3 && !COLLIE_SIMD_WITH_SSSE3 && !COLLIE_SIMD_WITH_SSE4_1 && !COLLIE_SIMD_WITH_SSE4_2 && !COLLIE_SIMD_WITH_AVX && !COLLIE_SIMD_WITH_AVX2 && !COLLIE_SIMD_WITH_AVXVNNI && !COLLIE_SIMD_WITH_FMA3_SSE && !COLLIE_SIMD_WITH_FMA4 && !COLLIE_SIMD_WITH_FMA3_AVX && !COLLIE_SIMD_WITH_FMA3_AVX2 && !COLLIE_SIMD_AVX512F && !COLLIE_SIMD_AVX512CD && !COLLIE_SIMD_AVX512DQ && !COLLIE_SIMD_AVX512BW && !COLLIE_SIMD_AVX512ER && !COLLIE_SIMD_AVX512PF && !COLLIE_SIMD_AVX512IFMA && !COLLIE_SIMD_AVX512VBMI && !COLLIE_SIMD_WITH_NEON && !COLLIE_SIMD_WITH_NEON64 && !COLLIE_SIMD_WITH_SVE && !COLLIE_SIMD_WITH_RVV && !COLLIE_SIMD_WASM
#define COLLIE_SIMD_NO_SUPPORTED_ARCHITECTURE
#endif

#endif  // COLLIE_PORT_SIMD_CONFIG_H_
