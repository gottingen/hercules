//
// Copyright 2020 The Turbo Authors.
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

#ifndef TURBO_PLATFORM_CONFIG_CONFIG_ASSERT_H_
#define TURBO_PLATFORM_CONFIG_CONFIG_ASSERT_H_

#include <cassert>
#include <cstdio>
#include <exception>
#include "turbo/platform/config/attribute_variable.h"
#include "turbo/platform/config/attribute_optimization.h"
#include "turbo/platform/config/config_have.h"
#include <cstdlib>


// `TURBO_INTERNAL_IMMEDIATE_ABORT_IMPL()` aborts the program in the fastest
// possible way, with no attempt at logging. One use is to implement hardening
// aborts with TURBO_OPTION_HARDENED.  Since this is an internal symbol, it
// should not be used directly outside of Turbo.
#if TURBO_HAVE_BUILTIN(__builtin_trap) || \
    (defined(__GNUC__) && !defined(__clang__))
#define TURBO_INTERNAL_IMMEDIATE_ABORT_IMPL() __builtin_trap()
#else
#define TURBO_INTERNAL_IMMEDIATE_ABORT_IMPL() abort()
#endif


// `TURBO_INTERNAL_UNREACHABLE_IMPL()` is the platform specific directive to
// indicate that a statement is unreachable, and to allow the compiler to
// optimize accordingly. Clients should use `TURBO_UNREACHABLE()`, which is
// defined below.
#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define TURBO_INTERNAL_UNREACHABLE_IMPL() std::unreachable()
#elif defined(__GNUC__) || TURBO_HAVE_BUILTIN(__builtin_unreachable)
#define TURBO_INTERNAL_UNREACHABLE_IMPL() __builtin_unreachable()
#elif TURBO_HAVE_BUILTIN(__builtin_assume)
#define TURBO_INTERNAL_UNREACHABLE_IMPL() __builtin_assume(false)
#elif defined(_MSC_VER)
#define TURBO_INTERNAL_UNREACHABLE_IMPL() __assume(false)
#else
#define TURBO_INTERNAL_UNREACHABLE_IMPL()
#endif

// `TURBO_INTERNAL_HARDENING_ABORT()` controls how `TURBO_HARDENING_ASSERT()`
// aborts the program in release mode (when NDEBUG is defined). The
// implementation should abort the program as quickly as possible and ideally it
// should not be possible to ignore the abort request.
#define TURBO_INTERNAL_HARDENING_ABORT()   \
  do {                                    \
    TURBO_INTERNAL_IMMEDIATE_ABORT_IMPL(); \
    TURBO_INTERNAL_UNREACHABLE_IMPL();     \
  } while (false)


// TURBO_ASSERT()
//
// In C++11, `assert` can't be used portably within constexpr functions.
// TURBO_ASSERT functions as a runtime assert but works in C++11 constexpr
// functions.  Example:
//
// constexpr double Divide(double a, double b) {
//   return TURBO_ASSERT(b != 0), a / b;
// }
//
// This macro is inspired by
// https://akrzemi1.wordpress.com/2017/05/18/asserts-in-constexpr-functions/
#if defined(NDEBUG)
#define TURBO_ASSERT(condition) \
  (false ? static_cast<void>(condition) : static_cast<void>(0))
#else
#define TURBO_ASSERT(expr)                           \
  (TURBO_LIKELY((expr)) ? static_cast<void>(0) \
                             : [] { assert(false && #expr); }())  // NOLINT
#endif

// TURBO_HARDENING_ASSERT()
//
// `TURBO_HARDENING_ASSERT()` is like `TURBO_ASSERT()`, but used to implement
// runtime assertions that should be enabled in hardened builds even when
// `NDEBUG` is defined.
//
// When `NDEBUG` is not defined, `TURBO_HARDENING_ASSERT()` is identical to
// `TURBO_ASSERT()`.
//
// See `TURBO_OPTION_HARDENED` in `turbo/base/options.h` for more information on
// hardened mode.
#if TURBO_OPTION_HARDENED == 1 && defined(NDEBUG)
#define TURBO_HARDENING_ASSERT(expr)                 \
  (TURBO_LIKELY((expr)) ? static_cast<void>(0) \
                             : [] { TURBO_INTERNAL_HARDENING_ABORT(); }())
#else
#define TURBO_HARDENING_ASSERT(expr) TURBO_ASSERT(expr)
#endif


#ifdef TURBO_HAVE_EXCEPTIONS
#define TURBO_INTERNAL_TRY try
#define TURBO_INTERNAL_CATCH_ANY catch (...)
#define TURBO_INTERNAL_RETHROW do { throw; } while (false)
#else  // TURBO_HAVE_EXCEPTIONS
#define TURBO_INTERNAL_TRY if (true)
#define TURBO_INTERNAL_CATCH_ANY else if (false)
#define TURBO_INTERNAL_RETHROW do {} while (false)
#endif  // TURBO_HAVE_EXCEPTIONS


// `TURBO_UNREACHABLE()` is an unreachable statement.  A program which reaches
// one has undefined behavior, and the compiler may optimize accordingly.
#if TURBO_OPTION_HARDENED == 1 && defined(NDEBUG)
// Abort in hardened mode to avoid dangerous undefined behavior.
#define TURBO_UNREACHABLE()                \
  do {                                    \
    TURBO_INTERNAL_IMMEDIATE_ABORT_IMPL(); \
    TURBO_INTERNAL_UNREACHABLE_IMPL();     \
  } while (false)
#else
// The assert only fires in debug mode to aid in debugging.
// When NDEBUG is defined, reaching TURBO_UNREACHABLE() is undefined behavior.
#define TURBO_UNREACHABLE()                       \
  do {                                           \
    /* NOLINTNEXTLINE: misc-static-assert */     \
    assert(false && "TURBO_UNREACHABLE reached"); \
    TURBO_INTERNAL_UNREACHABLE_IMPL();            \
  } while (false)
#endif


// TURBO_ASSUME(cond)
//
// Informs the compiler that a condition is always true and that it can assume
// it to be true for optimization purposes.
//
// WARNING: If the condition is false, the program can produce undefined and
// potentially dangerous behavior.
//
// In !NDEBUG mode, the condition is checked with an assert().
//
// NOTE: The expression must not have side effects, as it may only be evaluated
// in some compilation modes and not others. Some compilers may issue a warning
// if the compiler cannot prove the expression has no side effects. For example,
// the expression should not use a function call since the compiler cannot prove
// that a function call does not have side effects.
//
// Example:
//
//   int x = ...;
//   TURBO_ASSUME(x >= 0);
//   // The compiler can optimize the division to a simple right shift using the
//   // assumption specified above.
//   int y = x / 16;
//
#if !defined(NDEBUG)
#define TURBO_ASSUME(cond) assert(cond)
#elif TURBO_HAVE_BUILTIN(__builtin_assume)
#define TURBO_ASSUME(cond) __builtin_assume(cond)
#elif defined(_MSC_VER)
#define TURBO_ASSUME(cond) __assume(cond)
#elif defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define TURBO_ASSUME(cond)            \
      do {                               \
        if (!(cond)) std::unreachable(); \
      } while (false)
#elif defined(__GNUC__) || TURBO_HAVE_BUILTIN(__builtin_unreachable)
#define TURBO_ASSUME(cond)                 \
      do {                                    \
        if (!(cond)) __builtin_unreachable(); \
      } while (false)
#else
#define TURBO_ASSUME(cond)               \
      do {                                  \
        static_cast<void>(false && (cond)); \
      } while (false)
#endif


// ------------------------------------------------------------------------
// TURBO_ANALYSIS_ASSUME
//
// This acts the same as the VC++ __analysis_assume directive and is implemented
// simply as a wrapper around it to allow portable usage of it and to take
// advantage of it if and when it appears in other compilers.
//
// Example usage:
//    char Function(char* p) {
//       TURBO_ANALYSIS_ASSUME(p != NULL);
//       return *p;
//    }
//
#ifndef TURBO_ANALYSIS_ASSUME
#if defined(_MSC_VER) && (_MSC_VER >= 1300) // If VC7.0 and later
#define TURBO_ANALYSIS_ASSUME(x) __analysis_assume(!!(x)) // !! because that allows for convertible-to-bool in addition to bool.
#else
#define TURBO_ANALYSIS_ASSUME(x)
#endif
#endif


#endif  // TURBO_PLATFORM_CONFIG_CONFIG_ASSERT_H_
