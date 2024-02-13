// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#if defined(__has_feature)
#define HERCULES_HAS_FEATURE(...) __has_feature(__VA_ARGS__)
#else
#define HERCULES_HAS_FEATURE(...) 0
#endif

/* Define a convenience macro to test when address sanitizer is being used
 * across the different compilers (e.g. clang, gcc) */
#if HERCULES_HAS_FEATURE(address_sanitizer) || __SANITIZE_ADDRESS__
#define HERCULES_SANITIZE_ADDRESS 1
#endif

/*! \brief whether or not use c++11 support */
#ifndef HERCULES_USE_CXX11
#if defined(__GXX_EXPERIMENTAL_CXX0X__) || defined(_MSC_VER)
#define HERCULES_USE_CXX11 1
#else
#define HERCULES_USE_CXX11 (__cplusplus >= 201103L)
#endif
#endif

/*! \brief strict CXX11 support */
#ifndef HERCULES_STRICT_CXX11
#if defined(_MSC_VER)
#define HERCULES_STRICT_CXX11 1
#else
#define HERCULES_STRICT_CXX11 (__cplusplus >= 201103L)
#endif
#endif

#if HERCULES_USE_CXX11
#define HERCULES_THROW_EXCEPTION noexcept(false)
#define HERCULES_NO_EXCEPTION noexcept(true)
#else
#define HERCULES_THROW_EXCEPTION
#define HERCULES_NO_EXCEPTION
#endif

/*! \brief Whether cxx11 thread local is supported */
#ifndef HERCULES_CXX11_THREAD_LOCAL
#if defined(_MSC_VER)
#define HERCULES_CXX11_THREAD_LOCAL (_MSC_VER >= 1900)
#elif defined(__clang__)
#define HERCULES_CXX11_THREAD_LOCAL (HERCULES_HAS_FEATURE(cxx_thread_local))
#else
#define HERCULES_CXX11_THREAD_LOCAL (__cplusplus >= 201103L)
#endif
#endif

/*! \brief Whether to use modern thread local construct */
#ifndef HERCULES_MODERN_THREAD_LOCAL
#define HERCULES_MODERN_THREAD_LOCAL 1
#endif

/// check if g++ is before 4.6
#if HERCULES_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 6
#pragma message(                                 \
    "Will need g++-4.6 or higher to compile all" \
    "the features in hvm-runtime, "             \
    "compile without c++0x, some features may be disabled")
#undef HERCULES_USE_CXX11
#define HERCULES_USE_CXX11 0
#endif
#endif

/**
 * "Cold" indicates to the compiler that a function is only expected to be
 * called from unlikely code paths. It can affect decisions made by the
 * optimizer both when processing the function body and when analyzing
 * call-sites.
 */
#if __GNUC__
#define HERCULES_COLD __attribute__((__cold__))
#else
#define HERCULES_COLD
#endif

// always inline
#ifdef _MSC_VER
#define HERCULES_ALWAYS_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
#define HERCULES_ALWAYS_INLINE inline __attribute__((__always_inline__))
#else
#define HERCULES_ALWAYS_INLINE inline
#endif

// noinline
#if defined(_MSC_VER)
#define HERCULES_NO_INLINE __declspec(noinline)
#else
#define HERCULES_NO_INLINE __attribute__((noinline))
#endif

// attribute hidden
#if defined(_MSC_VER)
#define HERCULES_ATTR_VISIBILITY_HIDDEN
#elif defined(__GNUC__)
#define HERCULES_ATTR_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define HERCULES_ATTR_VISIBILITY_HIDDEN
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define HERCULES_DLL EMSCRIPTEN_KEEPALIVE
#endif

#ifndef HERCULES_DLL
#ifdef _WIN32
#ifdef HERCULES_EXPORTS
#define HERCULES_DLL __declspec(dllexport)
#else
#define HERCULES_DLL __declspec(dllimport)
#endif
#else
#define HERCULES_DLL __attribute__((visibility("default")))
#endif
#endif

#define HERCULES_INLINE_VISIBILITY HERCULES_ATTR_VISIBILITY_HIDDEN HERCULES_ALWAYS_INLINE

/*! \brief helper macro to supress unused warning */
#if defined(__GNUC__)
#define HERCULES_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define HERCULES_ATTRIBUTE_UNUSED
#endif

// warn unused result
#if defined(_MSC_VER) && (_MSC_VER >= 1700)
#define HERCULES_WARN_UNUSED_RESULT _Check_return_
#elif defined(__clang__) || defined(__GNUC__)
#define HERCULES_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#else
#define HERCULES_WARN_UNUSED_RESULT
#endif

/*! \brief helper macro to supress Undefined Behavior Sanitizer for a specific function */
#if defined(__clang__)
#define HERCULES_SUPPRESS_UBSAN __attribute__((no_sanitize("undefined")))
#elif defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ >= 409)
#define HERCULES_SUPPRESS_UBSAN __attribute__((no_sanitize_undefined))
#else
#define HERCULES_SUPPRESS_UBSAN
#endif

// __ubsan_xxx is copy from pytorch
// https://github.com/pytorch/pytorch/blob/release/1.11/c10/macros/Macros.h
#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__ __attribute__((no_sanitize("signed-integer-overflow")))
#define __ubsan_ignore_function__ __attribute__((no_sanitize("function")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#define __ubsan_ignore_function__
#endif

// Generalize warning push/pop.
#if defined(__GNUC__) || defined(__clang__)
// Clang & GCC
#define HERCULES_PUSH_WARNING _Pragma("GCC diagnostic push")
#define HERCULES_POP_WARNING _Pragma("GCC diagnostic pop")
#define HERCULES_GNU_DISABLE_WARNING_INTERNAL2(warningName) #warningName
#define HERCULES_GNU_DISABLE_WARNING(warningName) \
  _Pragma(HERCULES_GNU_DISABLE_WARNING_INTERNAL2(GCC diagnostic ignored warningName))
#ifdef __clang__
#define HERCULES_CLANG_DISABLE_WARNING(warningName) HERCULES_GNU_DISABLE_WARNING(warningName)
#define HERCULES_GCC_DISABLE_WARNING(warningName)
#else
#define HERCULES_CLANG_DISABLE_WARNING(warningName)
#define HERCULES_GCC_DISABLE_WARNING(warningName) HERCULES_GNU_DISABLE_WARNING(warningName)
#endif
#define HERCULES_MSVC_DISABLE_WARNING(warningNumber)
#elif defined(_MSC_VER)
#define HERCULES_PUSH_WARNING __pragma(warning(push))
#define HERCULES_POP_WARNING __pragma(warning(pop))
// Disable the GCC warnings.
#define HERCULES_GNU_DISABLE_WARNING(warningName)
#define HERCULES_GCC_DISABLE_WARNING(warningName)
#define HERCULES_CLANG_DISABLE_WARNING(warningName)
#define HERCULES_MSVC_DISABLE_WARNING(warningNumber) __pragma(warning(disable : warningNumber))
#else
#define HERCULES_PUSH_WARNING
#define HERCULES_POP_WARNING
#define HERCULES_GNU_DISABLE_WARNING(warningName)
#define HERCULES_GCC_DISABLE_WARNING(warningName)
#define HERCULES_CLANG_DISABLE_WARNING(warningName)
#define HERCULES_MSVC_DISABLE_WARNING(warningNumber)
#endif

#ifdef __clang__
#define _HERCULES_PRAGMA__(string) _Pragma(#string)
#define _HERCULES_PRAGMA_(string) _HERCULES_PRAGMA__(string)
#define HERCULES_CLANG_DIAGNOSTIC_PUSH() _Pragma("clang diagnostic push")
#define HERCULES_CLANG_DIAGNOSTIC_POP() _Pragma("clang diagnostic pop")
#define HERCULES_CLANG_DIAGNOSTIC_IGNORE(flag) _HERCULES_PRAGMA_(clang diagnostic ignored flag)
#define HERCULES_CLANG_HAS_WARNING(flag) __has_warning(flag)
#else
#define HERCULES_CLANG_DIAGNOSTIC_PUSH()
#define HERCULES_CLANG_DIAGNOSTIC_POP()
#define HERCULES_CLANG_DIAGNOSTIC_IGNORE(flag)
#define HERCULES_CLANG_HAS_WARNING(flag) 0
#endif

#ifdef HERCULES_HAVE_SHADOW_LOCAL_WARNINGS
#define HERCULES_GCC_DISABLE_NEW_SHADOW_WARNINGS            \
  HERCULES_GNU_DISABLE_WARNING("-Wshadow-compatible-local") \
  HERCULES_GNU_DISABLE_WARNING("-Wshadow-local")            \
  HERCULES_GNU_DISABLE_WARNING("-Wshadow")
#else
#define HERCULES_GCC_DISABLE_NEW_SHADOW_WARNINGS /* empty */
#endif

//  and to force the compiler to optimize for the fast path, even when it is not
//  overwhelmingly likely.
#if __GNUC__
#define HERCULES_DETAIL_BUILTIN_EXPECT(b, t) (__builtin_expect(b, t))
#else
#define HERCULES_DETAIL_BUILTIN_EXPECT(b, t) b
#endif
#define HERCULES_LIKELY(x) HERCULES_DETAIL_BUILTIN_EXPECT((x), 1)
#define HERCULES_UNLIKELY(x) HERCULES_DETAIL_BUILTIN_EXPECT((x), 0)

// HERCULES_ASSERT()
//
// In C++11, `assert` can't be used portably within constexpr functions.
// HERCULES_ASSERT functions as a runtime assert but works in C++11 constexpr
// functions.  Example:
//
// constexpr double Divide(double a, double b) {
//   return HERCULES_ASSERT(b != 0), a / b;
// }
//
// This macro is inspired by
// https://akrzemi1.wordpress.com/2017/05/18/asserts-in-constexpr-functions/
#if defined(NDEBUG)
#define HERCULES_ASSERT(expr) (false ? static_cast<void>(expr) : static_cast<void>(0))
#else
#define HERCULES_ASSERT(expr) \
  (HERCULES_LIKELY((expr)) ? static_cast<void>(0) : [] { assert(false && #expr); }())
#endif

/*!
 * \brief whether throw ::hercules::runtime::Error instead of
 *  directly calling abort when FATAL error occurred
 *  NOTE: this may still not be perfect.
 *  do not use FATAL and CHECK in destructors
 */
#ifndef HERCULES_LOG_FATAL_THROW
#define HERCULES_LOG_FATAL_THROW 1
#endif

/*!
 * \brief whether always log a message before throw
 * This can help identify the error that cannot be catched.
 */
#ifndef HERCULES_LOG_BEFORE_THROW
#define HERCULES_LOG_BEFORE_THROW 0
#endif

/*!
 * \brief Whether to use customized logger,
 * whose output can be decided by other libraries.
 */
#ifndef HERCULES_LOG_CUSTOMIZE
#define HERCULES_LOG_CUSTOMIZE 0
#endif

/*!
 * \brief Whether to enable debug logging feature.
 */
#ifndef HERCULES_LOG_DEBUG
#ifdef NDEBUG
#define HERCULES_LOG_DEBUG 0
#else
#define HERCULES_LOG_DEBUG 1
#endif
#endif

/*!
 * \brief Whether to disable date message on the log.
 */
#ifndef HERCULES_LOG_NODATE
#define HERCULES_LOG_NODATE 0
#endif

/*! \brief helper macro to generate string concat */
#define HERCULES_STR_CONCAT_(__x, __y) __x##__y
#define HERCULES_STR_CONCAT(__x, __y) HERCULES_STR_CONCAT_(__x, __y)
#define HERCULES_AS_STR_(x) #x
#define HERCULES_AS_STR(x) HERCULES_AS_STR_(x)

/*!
 * \brief Disable copy constructor and assignment operator.
 *
 * If C++11 is supported, both copy and move constructors and
 * assignment operators are deleted explicitly. Otherwise, they are
 * only declared but not implemented. Place this macro in private
 * section if C++11 is not available.
 */
#ifndef DISALLOW_COPY_AND_ASSIGN
#if HERCULES_USE_CXX11
#define DISALLOW_COPY_AND_ASSIGN(T) \
  T(T const&) = delete;             \
  T(T&&) = delete;                  \
  T& operator=(T const&) = delete;  \
  T& operator=(T&&) = delete
#else
#define DISALLOW_COPY_AND_ASSIGN(T) \
  T(T const&);                      \
  T& operator=(T const&)
#endif
#endif

/*
 * \brief Define the default copy/move constructor and assign operator
 * \param TypeName The class typename.
 */
#define HERCULES_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName) \
  TypeName(const TypeName& other) = default;                     \
  TypeName(TypeName&& other) noexcept = default;                 \
  TypeName& operator=(const TypeName& other) = default;          \
  TypeName& operator=(TypeName&& other) noexcept = default;

#include <string>
#include <vector>

#if defined(__cpp_lib_experimental_string_view) && __cpp_lib_experimental_string_view >= 201411
#define HERCULES_USE_CXX14_STRING_VIEW 1
#else
#define HERCULES_USE_CXX14_STRING_VIEW 0
#endif

// Tested with clang version 9.0.1 and c++17. It will detect string_view support
// correctly.
#if defined(__cpp_lib_string_view) && __cpp_lib_string_view >= 201606
#define HERCULES_USE_CXX17_STRING_VIEW 1
#else
#define HERCULES_USE_CXX17_STRING_VIEW 0
#endif

// align function
#if defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
#define HERCULES_ALIGN_FUNCTION __attribute__((aligned(128)))
#elif defined _MSC_VER
#define HERCULES_ALIGN_FUNCTION
#else
#define HERCULES_ALIGN_FUNCTION
#endif

// align address
#ifndef HERCULES_MEMORY_ALIGNMENT
#define HERCULES_MEMORY_ALIGNMENT sizeof(unsigned long) /* platform word */
#endif

#define hercules_memory_align(d, a) (((d) + ((a)-1)) & ~((a)-1))
#define hercules_memory_align_ptr(p, a) \
  (unsigned char*)(((uintptr_t)(p) + ((uintptr_t)(a)-1)) & ~((uintptr_t)(a)-1))

namespace hercules {
namespace runtime {

// Endianness
#ifdef _MSC_VER
// It's MSVC, so we just have to guess ... and allow an override
#ifdef HERCULES_ENDIAN_BE
constexpr auto kIsLittleEndian = false;
#else
constexpr auto kIsLittleEndian = true;
#endif
#else
constexpr auto kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#endif
constexpr auto kIsBigEndian = !kIsLittleEndian;

HERCULES_ALWAYS_INLINE void assume(bool cond) {
#if defined(__clang__)  // Must go first because Clang also defines __GNUC__.
  __builtin_assume(cond);
#elif defined(__GNUC__)
  if (!cond) {
    __builtin_unreachable();
  }
#elif defined(_MSC_VER)
  __assume(cond);
#else
  // Do nothing.
#endif
}

HERCULES_ALWAYS_INLINE void assume_unreachable() {
  assume(false);
  // Do a bit more to get the compiler to understand
  // that this function really will never return.
#if defined(__GNUC__)
  __builtin_unreachable();
#elif defined(_MSC_VER)
  __assume(0);
#else
  // Well, it's better than nothing.
  std::abort();
#endif
}

/*!
 * \brief safely get the beginning address of a vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template <typename T>
inline T* BeginPtr(std::vector<T>& vec) {  // NOLINT(*)
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a const vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template <typename T>
inline const T* BeginPtr(const std::vector<T>& vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a string
 * \param str input string
 * \return beginning address of a string
 */
inline char* BeginPtr(std::string& str) {  // NOLINT(*)
  if (str.length() == 0)
    return NULL;
  return &str[0];
}
/*!
 * \brief get the beginning address of a const string
 * \param str input string
 * \return beginning address of a string
 */
inline const char* BeginPtr(const std::string& str) {
  if (str.length() == 0)
    return NULL;
  return &str[0];
}

}  // namespace runtime
}  // namespace hercules

/*
 * The following code is taken from https://github.com/python/cpython/blob/main/Include/pyport.h
 * It is mainly used for py_unicode_object py_unicodedata
 */

#if (defined(__GNUC__) && !defined(__STRICT_ANSI__) && \
     (((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1)) || (__GNUC__ >= 4)))
static_assert(false, "need gcc 4+");
#else
#define HERCULES_ARRAY_LENGTH(array) (sizeof(array) / sizeof((array)[0]))
#endif

/* Largest positive value of type tx_ssize_t. */
#define HERCULES_SSIZE_T_MAX ((ssize_t)(((size_t)-1) >> 1))
/* Smallest negative value of type tx_ssize_t. */
#define HERCULES_SSIZE_T_MIN (-HERCULES_SSIZE_T_MAX - 1)

#define HERCULES_SAFE_DOWNCAST(VALUE, WIDE, NARROW) (NARROW)(VALUE)
