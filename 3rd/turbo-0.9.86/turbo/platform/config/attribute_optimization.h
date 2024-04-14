//
// Copyright 2023 The Turbo Authors.
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
//
// -----------------------------------------------------------------------------
// File: optimization.h
// -----------------------------------------------------------------------------
//
// This header file defines portable macros for performance optimization.

#ifndef TURBO_PLATFORM_ATTRIBUTE_OPTIMIZATION_H_
#define TURBO_PLATFORM_ATTRIBUTE_OPTIMIZATION_H_

#include <cassert>

#include "turbo/platform/config/compiler_traits.h"
#include "turbo/platform/config/attribute_warning.h"
#include "turbo/platform/options.h"

// TURBO_BLOCK_TAIL_CALL_OPTIMIZATION
//
// Instructs the compiler to avoid optimizing tail-call recursion. This macro is
// useful when you wish to preserve the existing function order within a stack
// trace for logging, debugging, or profiling purposes.
//
// Example:
//
//   int f() {
//     int result = g();
//     TURBO_BLOCK_TAIL_CALL_OPTIMIZATION();
//     return result;
//   }
#if defined(__pnacl__)
#define TURBO_BLOCK_TAIL_CALL_OPTIMIZATION() if (volatile int x = 0) { (void)x; }
#elif defined(__clang__)
// Clang will not tail call given inline volatile assembly.
#define TURBO_BLOCK_TAIL_CALL_OPTIMIZATION() __asm__ __volatile__("")
#elif defined(__GNUC__)
// GCC will not tail call given inline volatile assembly.
#define TURBO_BLOCK_TAIL_CALL_OPTIMIZATION() __asm__ __volatile__("")
#elif defined(_MSC_VER)
#include <intrin.h>
// The __nop() intrinsic blocks the optimisation.
#define TURBO_BLOCK_TAIL_CALL_OPTIMIZATION() __nop()
#else
#define TURBO_BLOCK_TAIL_CALL_OPTIMIZATION() if (volatile int x = 0) { (void)x; }
#endif

// TURBO_CACHE_LINE_ALIGNED
//
// Indicates that the declared object be cache aligned using
// `TURBO_CACHE_LINE_SIZE` (see above). Cacheline aligning objects allows you to
// load a set of related objects in the L1 cache for performance improvements.
// Cacheline aligning objects properly allows constructive memory sharing and
// prevents destructive (or "false") memory sharing.
//
// NOTE: callers should replace uses of this macro with `alignas()` using
// `std::hardware_constructive_interference_size` and/or
// `std::hardware_destructive_interference_size` when C++17 becomes available to
// them.
//
// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0154r1.html
// for more information.
//
// On some compilers, `TURBO_CACHE_LINE_ALIGNED` expands to an `__attribute__`
// or `__declspec` attribute. For compilers where this is not known to work,
// the macro expands to nothing.
//
// No further guarantees are made here. The result of applying the macro
// to variables and types is always implementation-defined.
//
// WARNING: It is easy to use this attribute incorrectly, even to the point
// of causing bugs that are difficult to diagnose, crash, etc. It does not
// of itself guarantee that objects are aligned to a cache line.
//
// NOTE: Some compilers are picky about the locations of annotations such as
// this attribute, so prefer to put it at the beginning of your declaration.
// For example,
//
//   TURBO_CACHE_LINE_ALIGNED static Foo* foo = ...
//
//   class TURBO_CACHE_LINE_ALIGNED Bar { ...
//
// Recommendations:
//
// 1) Consult compiler documentation; this comment is not kept in sync as
//    toolchains evolve.
// 2) Verify your use has the intended effect. This often requires inspecting
//    the generated machine code.
// 3) Prefer applying this attribute to individual variables. Avoid
//    applying it to types. This tends to localize the effect.
#if defined(__clang__) || defined(__GNUC__)
#define TURBO_CACHE_LINE_ALIGNED __attribute__((aligned(TURBO_CACHE_LINE_SIZE)))
#elif defined(_MSC_VER)
#define TURBO_CACHE_LINE_ALIGNED __declspec(align(TURBO_CACHE_LINE_SIZE))
#else
#define TURBO_CACHE_LINE_ALIGNED
#endif


// TURBO_INTERNAL_UNIQUE_SMALL_NAME(cond)
// This macro forces small unique name on a static file level symbols like
// static local variables or static functions. This is intended to be used in
// macro definitions to optimize the cost of generated code. Do NOT use it on
// symbols exported from translation unit since it may cause a link time
// conflict.
//
// Example:
//
// #define MY_MACRO(txt)
// namespace {
//  char VeryVeryLongVarName[] TURBO_INTERNAL_UNIQUE_SMALL_NAME() = txt;
//  const char* VeryVeryLongFuncName() TURBO_INTERNAL_UNIQUE_SMALL_NAME();
//  const char* VeryVeryLongFuncName() { return txt; }
// }
//

#if defined(__GNUC__)
#define TURBO_INTERNAL_UNIQUE_SMALL_NAME2(x) #x
#define TURBO_INTERNAL_UNIQUE_SMALL_NAME1(x) TURBO_INTERNAL_UNIQUE_SMALL_NAME2(x)
#define TURBO_INTERNAL_UNIQUE_SMALL_NAME() \
  asm(TURBO_INTERNAL_UNIQUE_SMALL_NAME1(.turbo.__COUNTER__))
#else
#define TURBO_INTERNAL_UNIQUE_SMALL_NAME()
#endif


// ------------------------------------------------------------------------
// TURBO_OPTIMIZE_OFF / TURBO_OPTIMIZE_ON
//
// Implements portable inline optimization enabling/disabling.
// Usage of these macros must be in order OFF then ON. This is
// because the OFF macro pushes a set of settings and the ON
// macro pops them. The nesting of OFF/ON sets (e.g. OFF, OFF, ON, ON)
// is not guaranteed to work on all platforms.
//
// This is often used to allow debugging of some code that's
// otherwise compiled with undebuggable optimizations. It's also
// useful for working around compiler code generation problems
// that occur in optimized builds.
//
// Some compilers (e.g. VC++) don't allow doing this within a function and
// so the usage must be outside a function, as with the example below.
// GCC on x86 appears to have some problem with argument passing when
// using TURBO_OPTIMIZE_OFF in optimized builds.
//
// Example usage:
//     // Disable optimizations for SomeFunction.
//     TURBO_OPTIMIZE_OFF()
//     void SomeFunction()
//     {
//         ...
//     }
//     TURBO_OPTIMIZE_ON()
//
#if !defined(TURBO_OPTIMIZE_OFF)
#if   defined(TURBO_COMPILER_MSVC)
#define TURBO_OPTIMIZE_OFF() __pragma(optimize("", off))
#elif defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION > 4004) && (defined(__i386__) || defined(__x86_64__)) // GCC 4.4+ - Seems to work only on x86/Linux so far. However, GCC 4.4 itself appears broken and screws up parameter passing conventions.
#define TURBO_OPTIMIZE_OFF()            \
				_Pragma("GCC push_options")      \
				_Pragma("GCC optimize 0")
#elif defined(TURBO_COMPILER_CLANG) && (!defined(TURBO_PLATFORM_ANDROID) || (TURBO_COMPILER_VERSION >= 380))
#define TURBO_OPTIMIZE_OFF() \
				TURBO_DISABLE_CLANG_WARNING(-Wunknown-pragmas) \
				_Pragma("clang optimize off") \
				TURBO_RESTORE_CLANG_WARNING()
#else
#define TURBO_OPTIMIZE_OFF()
#endif
#endif

#if !defined(TURBO_OPTIMIZE_ON)
#if   defined(TURBO_COMPILER_MSVC)
#define TURBO_OPTIMIZE_ON() __pragma(optimize("", on))
#elif defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION > 4004) && (defined(__i386__) || defined(__x86_64__)) // GCC 4.4+ - Seems to work only on x86/Linux so far. However, GCC 4.4 itself appears broken and screws up parameter passing conventions.
#define TURBO_OPTIMIZE_ON() _Pragma("GCC pop_options")
#elif defined(TURBO_COMPILER_CLANG) && (!defined(TURBO_PLATFORM_ANDROID) || (TURBO_COMPILER_VERSION >= 380))
#define TURBO_OPTIMIZE_ON() \
				TURBO_DISABLE_CLANG_WARNING(-Wunknown-pragmas) \
				_Pragma("clang optimize on") \
				TURBO_RESTORE_CLANG_WARNING()
#else
#define TURBO_OPTIMIZE_ON()
#endif
#endif

#endif  // TURBO_PLATFORM_ATTRIBUTE_OPTIMIZATION_H_
