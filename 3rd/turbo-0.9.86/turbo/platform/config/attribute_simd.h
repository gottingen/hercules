// Copyright 2023 The titan-search Authors.
//
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

#ifndef TURBO_SIMD_CONFIG_CONFIG_H_
#define TURBO_SIMD_CONFIG_CONFIG_H_

/**
 * high level free functions
 *
 * @defgroup turbo_config_macro Instruction Set Detection
 */

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if SSE2 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE2__
#define TURBO_WITH_SSE2 1
#else
#define TURBO_WITH_SSE2 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if SSE3 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE3__
#define TURBO_WITH_SSE3 1
#else
#define TURBO_WITH_SSE3 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if SSSE3 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSSE3__
#define TURBO_WITH_SSSE3 1
#else
#define TURBO_WITH_SSSE3 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if SSE4.1 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE4_1__
#define TURBO_WITH_SSE4_1 1
#else
#define TURBO_WITH_SSE4_1 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if SSE4.2 is available at compile-time, to 0 otherwise.
 */
#ifdef __SSE4_2__
#define TURBO_WITH_SSE4_2 1
#else
#define TURBO_WITH_SSE4_2 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if AVX is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX__
#define TURBO_WITH_AVX 1
#else
#define TURBO_WITH_AVX 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if AVX2 is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX2__
#define TURBO_WITH_AVX2 1
#else
#define TURBO_WITH_AVX2 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if FMA3 for SSE is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA__

#if defined(__SSE__)
#ifndef TURBO_WITH_FMA3_SSE // Leave the opportunity to manually disable it, see #643
#define TURBO_WITH_FMA3_SSE 1
#endif
#else

#if defined(TURBO_WITH_FMA3_SSE) && TURBO_WITH_FMA3_SSE
#error "Manually set TURBO_WITH_FMA3_SSE is incompatible with current compiler flags"
#endif

#define TURBO_WITH_FMA3_SSE 0
#endif

#else

#if defined(TURBO_WITH_FMA3_SSE) && TURBO_WITH_FMA3_SSE
#error "Manually set TURBO_WITH_FMA3_SSE is incompatible with current compiler flags"
#endif

#define TURBO_WITH_FMA3_SSE 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if FMA3 for AVX is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA__

#if defined(__AVX__)
#ifndef TURBO_WITH_FMA3_AVX // Leave the opportunity to manually disable it, see #643
#define TURBO_WITH_FMA3_AVX 1
#endif
#else

#if defined(TURBO_WITH_FMA3_AVX) && TURBO_WITH_FMA3_AVX
#error "Manually set TURBO_WITH_FMA3_AVX is incompatible with current compiler flags"
#endif

#define TURBO_WITH_FMA3_AVX 0
#endif

#if defined(__AVX2__)
#ifndef TURBO_WITH_FMA3_AVX2 // Leave the opportunity to manually disable it, see #643
#define TURBO_WITH_FMA3_AVX2 1
#endif
#else

#if defined(TURBO_WITH_FMA3_AVX2) && TURBO_WITH_FMA3_AVX2
#error "Manually set TURBO_WITH_FMA3_AVX2 is incompatible with current compiler flags"
#endif

#define TURBO_WITH_FMA3_AVX2 0
#endif

#else

#if defined(TURBO_WITH_FMA3_AVX) && TURBO_WITH_FMA3_AVX
#error "Manually set TURBO_WITH_FMA3_AVX is incompatible with current compiler flags"
#endif

#if defined(TURBO_WITH_FMA3_AVX2) && TURBO_WITH_FMA3_AVX2
#error "Manually set TURBO_WITH_FMA3_AVX2 is incompatible with current compiler flags"
#endif

#define TURBO_WITH_FMA3_AVX 0
#define TURBO_WITH_FMA3_AVX2 0

#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if FMA4 is available at compile-time, to 0 otherwise.
 */
#ifdef __FMA4__
#define TURBO_WITH_FMA4 1
#else
#define TURBO_WITH_FMA4 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if AVX512F is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512F__
// AVX512 instructions are supported starting with gcc 6
// see https://www.gnu.org/software/gcc/gcc-6/changes.html
// check clang first, newer clang always defines __GNUC__ = 4
#if defined(__clang__) && __clang_major__ >= 6
#define TURBO_WITH_AVX512F 1
#elif defined(__GNUC__) && __GNUC__ < 6
#define TURBO_WITH_AVX512F 0
#else
#define TURBO_WITH_AVX512F 1
#if __GNUC__ == 6
#define TURBO_AVX512_SHIFT_INTRINSICS_IMM_ONLY 1
#endif
#endif
#else
#define TURBO_WITH_AVX512F 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if AVX512CD is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512CD__
// Avoids repeating the GCC workaround over and over
#define TURBO_WITH_AVX512CD TURBO_WITH_AVX512F
#else
#define TURBO_WITH_AVX512CD 0
#endif


#ifndef TURBO_WITH_AVX512VL
#if defined(__AVX512VL__) && __AVX512VL__ == 1
#define TURBO_WITH_AVX512VL 1
#else
#define TURBO_WITH_AVX512VL 0
#endif
#endif


#ifndef TURBO_WITH_AVX512VBMI
# if defined(__AVX512VBMI__) && __AVX512VBMI__ == 1
#define TURBO_WITH_AVX512VBMI 1
#else
#define TURBO_WITH_AVX512VBMI 0
#endif
#endif


#ifndef TURBO_WITH_AVX512VBMI2
#if defined(__AVX512VBMI2__) && __AVX512VBMI2__ == 1
#define TURBO_WITH_AVX512VBMI2 1
#else
#define TURBO_WITH_AVX512VBMI2 0
#endif
#endif

#ifndef TURBO_WITH_AVX512VNNI
#if defined(__AVX512VNNI__) && __AVX512VNNI__ == 1
#define TURBO_WITH_AVX512VNNI 1
#else
#define TURBO_WITH_AVX512VNNI 0
#endif
#endif

#ifndef TURBO_WITH_AVX512BITALG
# if defined(__AVX512BITALG__) && __AVX512BITALG__ == 1
#define TURBO_WITH_AVX512BITALG 1
#else
#define TURBO_WITH_AVX512BITALG 0
# endif
#endif

#ifndef TURBO_WITH_AVX512IFMA
# if defined(__AVX512IFMA__) && __AVX512IFMA__ == 1
#   define TURBO_WITH_AVX512IFMA 1
#else
#define TURBO_WITH_AVX512IFMA 0
# endif
#endif

#ifndef TURBO_WITH_AVX512VPOPCNTDQ
#if defined(__AVX512VPOPCNTDQ__) && __AVX512VPOPCNTDQ__ == 1
#define TURBO_WITH_AVX512VPOPCNTDQ 1
#else
#define TURBO_WITH_AVX512VPOPCNTDQ 0
#endif
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if AVX512DQ is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512DQ__
#define TURBO_WITH_AVX512DQ TURBO_WITH_AVX512F
#else
#define TURBO_WITH_AVX512DQ 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if AVX512BW is available at compile-time, to 0 otherwise.
 */
#ifdef __AVX512BW__
#define TURBO_WITH_AVX512BW TURBO_WITH_AVX512F
#else
#define TURBO_WITH_AVX512BW 0
#endif

#ifdef __ARM_NEON

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if NEON is available at compile-time, to 0 otherwise.
 */
#if __ARM_ARCH >= 7
#define TURBO_WITH_NEON 1
#else
#define TURBO_WITH_NEON 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if NEON64 is available at compile-time, to 0 otherwise.
 */
#ifdef __aarch64__
#define TURBO_WITH_NEON64 1
#else
#define TURBO_WITH_NEON64 0
#endif
#else
#define TURBO_WITH_NEON 0
#define TURBO_WITH_NEON64 0
#endif

/**
 * @ingroup turbo_config_macro
 *
 * Set to 1 if SVE is available and bit width is pre-set at compile-time, to 0 otherwise.
 */
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS > 0
#define TURBO_WITH_SVE 1
#define TURBO_SVE_BITS __ARM_FEATURE_SVE_BITS
#else
#define TURBO_WITH_SVE 0
#define TURBO_SVE_BITS 0
#endif

// Workaround for MSVC compiler
#ifdef _MSC_VER

#if TURBO_WITH_AVX512

#undef TURBO_WITH_AVX2
#define TURBO_WITH_AVX2 1

#endif

#if TURBO_WITH_AVX2

#undef TURBO_WITH_AVX
#define TURBO_WITH_AVX 1

#undef TURBO_WITH_FMA3_AVX
#define TURBO_WITH_FMA3_AVX 1

#undef TURBO_WITH_FMA3_AVX2
#define TURBO_WITH_FMA3_AVX2 1

#endif

#if TURBO_WITH_AVX

#undef TURBO_WITH_SSE4_2
#define TURBO_WITH_SSE4_2 1

#endif

#if TURBO_WITH_SSE4_2

#undef TURBO_WITH_SSE4_1
#define TURBO_WITH_SSE4_1 1

#endif

#if TURBO_WITH_SSE4_1

#undef TURBO_WITH_SSSE3
#define TURBO_WITH_SSSE3 1

#endif

#if TURBO_WITH_SSSE3

#undef TURBO_WITH_SSE3
#define TURBO_WITH_SSE3 1

#endif

#if TURBO_WITH_SSE3 || defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#undef TURBO_WITH_SSE2
#define TURBO_WITH_SSE2 1
#endif

#endif

#if !TURBO_WITH_SSE2 && !TURBO_WITH_SSE3 && !TURBO_WITH_SSSE3 && !TURBO_WITH_SSE4_1 && !TURBO_WITH_SSE4_2 && !TURBO_WITH_AVX && !TURBO_WITH_AVX2 && !TURBO_WITH_FMA3_SSE && !TURBO_WITH_FMA4 && !TURBO_WITH_FMA3_AVX && !TURBO_WITH_FMA3_AVX2 && !TURBO_WITH_AVX512F && !TURBO_WITH_AVX512CD && !TURBO_WITH_AVX512DQ && !TURBO_WITH_AVX512BW && !TURBO_WITH_NEON && !TURBO_WITH_NEON64 && !TURBO_WITH_SVE
#define TURBO_NO_SUPPORTED_ARCHITECTURE
#endif

#endif  // TURBO_SIMD_CONFIG_CONFIG_H_

