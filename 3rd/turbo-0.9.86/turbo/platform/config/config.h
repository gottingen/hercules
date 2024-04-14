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
//
// -----------------------------------------------------------------------------
// File: config.h
// -----------------------------------------------------------------------------
//
// This header file defines a set of macros for checking the presence of
// important compiler and platform features. Such macros can be used to
// produce portable code by parameterizing compilation based on the presence or
// lack of a given feature.
//
// We define a "feature" as some interface we wish to program to: for example,
// a library function or system call. A value of `1` indicates support for
// that feature; any other value indicates the feature support is undefined.
//
// Example:
//
// Suppose a programmer wants to write a program that uses the 'mmap()' system
// call. The Turbo macro for that feature (`TURBO_HAVE_MMAP`) allows you to
// selectively include the `mmap.h` header and bracket code using that feature
// in the macro:
//
//   #include "turbo/platform/config.h"
//
//   #ifdef TURBO_HAVE_MMAP
//   #include "sys/mman.h"
//   #endif  //TURBO_HAVE_MMAP
//
//   ...
//   #ifdef TURBO_HAVE_MMAP
//   void *ptr = mmap(...);
//   ...
//   #endif  // TURBO_HAVE_MMAP

#ifndef TURBO_PALTFORM_CONFIG_H_
#define TURBO_PALTFORM_CONFIG_H_

// Included for the __GLIBC__ macro (or similar macros on other systems).
#include <limits.h>

#ifdef __cplusplus
// Included for __GLIBCXX__, _LIBCPP_VERSION
#include <cstddef>
#endif  // __cplusplus

// TURBO_INTERNAL_CPLUSPLUS_LANG
//
// MSVC does not set the value of __cplusplus correctly, but instead uses
// _MSVC_LANG as a stand-in.
// https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros
//
// However, there are reports that MSVC even sets _MSVC_LANG incorrectly at
// times, for example:
// https://github.com/microsoft/vscode-cpptools/issues/1770
// https://reviews.llvm.org/D70996
//
// For this reason, this symbol is considered INTERNAL and code outside of
// Turbo must not use it.
#if defined(_MSVC_LANG)
#define TURBO_INTERNAL_CPLUSPLUS_LANG _MSVC_LANG
#elif defined(__cplusplus)
#define TURBO_INTERNAL_CPLUSPLUS_LANG __cplusplus
#endif

#if defined(__APPLE__)
// Included for TARGET_OS_IPHONE, __IPHONE_OS_VERSION_MIN_REQUIRED,
// __IPHONE_8_0.
#include <Availability.h>
#include <TargetConditionals.h>
#endif

#include "turbo/platform/config/policy_checks.h"
#include "turbo/platform/config/config_types.h"
#include "turbo/platform/config/config_have.h"
#include "turbo/platform/options.h"

// Turbo long-term support (LTS) releases will define
// `TURBO_LTS_RELEASE_VERSION` to the integer representing the date string of the
// LTS release version, and will define `TURBO_LTS_RELEASE_PATCH_LEVEL` to the
// integer representing the patch-level for that release.
//
// For example, for LTS release version "20300401.2", this would give us
// TURBO_LTS_RELEASE_VERSION == 20300401 && TURBO_LTS_RELEASE_PATCH_LEVEL == 2
//
// These symbols will not be defined in non-LTS code.
//
// Turbo recommends that clients live-at-head. Therefore, if you are using
// these symbols to assert a minimum version requirement, we recommend you do it
// as
//
// #if defined(TURBO_LTS_RELEASE_VERSION) && TURBO_LTS_RELEASE_VERSION < 20300401
// #error Project foo requires Turbo LTS version >= 20300401
// #endif
//
// The `defined(TURBO_LTS_RELEASE_VERSION)` part of the check excludes
// live-at-head clients from the minimum version assertion.
// LTS releases can be obtained from
#undef TURBO_LTS_RELEASE_VERSION
#undef TURBO_LTS_RELEASE_PATCH_LEVEL

// Helper macro to convert a CPP variable to a string literal.
#define TURBO_INTERNAL_DO_TOKEN_STR(x) #x
#define TURBO_INTERNAL_TOKEN_STR(x) TURBO_INTERNAL_DO_TOKEN_STR(x)

// TURBO_IS_LITTLE_ENDIAN
// TURBO_IS_BIG_ENDIAN
//
// Checks the endianness of the platform.
//
// Notes: uses the built in endian macros provided by GCC (since 4.6) and
// Clang (since 3.2); see
// https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html.
// Otherwise, if _WIN32, assume little endian. Otherwise, bail with an error.
#if defined(TURBO_IS_BIG_ENDIAN)
#error "TURBO_IS_BIG_ENDIAN cannot be directly set."
#endif
#if defined(TURBO_IS_LITTLE_ENDIAN)
#error "TURBO_IS_LITTLE_ENDIAN cannot be directly set."
#endif

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    #define TURBO_IS_LITTLE_ENDIAN 1
    #define TURBO_IS_BIG_ENDIAN 0
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define TURBO_IS_LITTLE_ENDIAN 0
#define TURBO_IS_BIG_ENDIAN 1
#elif defined(_WIN32)
#define TURBO_IS_LITTLE_ENDIAN 1
#define TURBO_IS_BIG_ENDIAN 0
#else
#error "turbo endian detection needs to be set up for your compiler"
#endif


// In debug mode, MSVC 2017's std::variant throws a EXCEPTION_ACCESS_VIOLATION
// SEH exception from emplace for variant<SomeStruct> when constructing the
// struct can throw. This defeats some of variant_test and
// variant_exception_safety_test.
#if defined(_MSC_VER) && _MSC_VER >= 1700 && defined(_DEBUG)
#define TURBO_INTERNAL_MSVC_2017_DBG_MODE
#endif

// TURBO_INTERNAL_MANGLED_NS
// TURBO_INTERNAL_MANGLED_BACKREFERENCE
//
// Internal macros for building up mangled names in our internal fork of CCTZ.
// This implementation detail is only needed and provided for the MSVC build.
//
// These macros both expand to string literals.  TURBO_INTERNAL_MANGLED_NS is
// the mangled spelling of the `turbo` namespace, and
// TURBO_INTERNAL_MANGLED_BACKREFERENCE is a back-reference integer representing
// the proper count to skip past the CCTZ fork namespace names.  (This number
// is one larger when there is an inline namespace name to skip.)
#if defined(_MSC_VER)
#if TURBO_OPTION_USE_INLINE_NAMESPACE == 0
#define TURBO_INTERNAL_MANGLED_NS "turbo"
#define TURBO_INTERNAL_MANGLED_BACKREFERENCE "5"
#else
#define TURBO_INTERNAL_MANGLED_NS \
  TURBO_INTERNAL_TOKEN_STR(TURBO_OPTION_INLINE_NAMESPACE_NAME) "@turbo"
#define TURBO_INTERNAL_MANGLED_BACKREFERENCE "6"
#endif
#endif

// TURBO_HAVE_MEMORY_SANITIZER
//
// MemorySanitizer (MSan) is a detector of uninitialized reads. It consists of
// a compiler instrumentation module and a run-time library.
#ifdef TURBO_HAVE_MEMORY_SANITIZER
#error "TURBO_HAVE_MEMORY_SANITIZER cannot be directly set."
#elif !defined(__native_client__) && TURBO_HAVE_FEATURE(memory_sanitizer)
#define TURBO_HAVE_MEMORY_SANITIZER 1
#endif

// TURBO_HAVE_THREAD_SANITIZER
//
// ThreadSanitizer (TSan) is a fast data race detector.
#ifdef TURBO_HAVE_THREAD_SANITIZER
#error "TURBO_HAVE_THREAD_SANITIZER cannot be directly set."
#elif defined(__SANITIZE_THREAD__)
#define TURBO_HAVE_THREAD_SANITIZER 1
#elif TURBO_HAVE_FEATURE(thread_sanitizer)
#define TURBO_HAVE_THREAD_SANITIZER 1
#endif

// TURBO_HAVE_ADDRESS_SANITIZER
//
// AddressSanitizer (ASan) is a fast memory error detector.
#ifdef TURBO_HAVE_ADDRESS_SANITIZER
#error "TURBO_HAVE_ADDRESS_SANITIZER cannot be directly set."
#elif defined(__SANITIZE_ADDRESS__)
#define TURBO_HAVE_ADDRESS_SANITIZER 1
#elif TURBO_HAVE_FEATURE(address_sanitizer)
#define TURBO_HAVE_ADDRESS_SANITIZER 1
#endif

// TURBO_HAVE_HWADDRESS_SANITIZER
//
// Hardware-Assisted AddressSanitizer (or HWASAN) is even faster than asan
// memory error detector which can use CPU features like ARM TBI, Intel LAM or
// AMD UAI.
#ifdef TURBO_HAVE_HWADDRESS_SANITIZER
#error "TURBO_HAVE_HWADDRESS_SANITIZER cannot be directly set."
#elif defined(__SANITIZE_HWADDRESS__)
#define TURBO_HAVE_HWADDRESS_SANITIZER 1
#elif TURBO_HAVE_FEATURE(hwaddress_sanitizer)
#define TURBO_HAVE_HWADDRESS_SANITIZER 1
#endif

// `TURBO_INTERNAL_HAS_RTTI` determines whether turbo is being compiled with
// RTTI support.
#ifdef TURBO_INTERNAL_HAS_RTTI
#error TURBO_INTERNAL_HAS_RTTI cannot be directly set
#elif TURBO_HAVE_FEATURE(cxx_rtti)
#define TURBO_INTERNAL_HAS_RTTI 1
#elif defined(__GNUC__) && defined(__GXX_RTTI)
#define TURBO_INTERNAL_HAS_RTTI 1
#elif defined(_MSC_VER) && defined(_CPPRTTI)
#define TURBO_INTERNAL_HAS_RTTI 1
#elif !defined(__GNUC__) && !defined(_MSC_VER)
// Unknown compiler, default to RTTI
#define TURBO_INTERNAL_HAS_RTTI 1
#endif

// TURBO_HAVE_LEAK_SANITIZER
//
// LeakSanitizer (or lsan) is a detector of memory leaks.
// https://clang.llvm.org/docs/LeakSanitizer.html
// https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer
//
// The macro TURBO_HAVE_LEAK_SANITIZER can be used to detect at compile-time
// whether the LeakSanitizer is potentially available. However, just because the
// LeakSanitizer is available does not mean it is active. Use the
// always-available run-time interface in //turbo/debugging/leak_check.h for
// interacting with LeakSanitizer.
#ifdef TURBO_HAVE_LEAK_SANITIZER
#error "TURBO_HAVE_LEAK_SANITIZER cannot be directly set."
#elif defined(LEAK_SANITIZER)
// GCC provides no method for detecting the presense of the standalone
// LeakSanitizer (-fsanitize=leak), so GCC users of -fsanitize=leak should also
// use -DLEAK_SANITIZER.
#define TURBO_HAVE_LEAK_SANITIZER 1
// Clang standalone LeakSanitizer (-fsanitize=leak)
#elif TURBO_HAVE_FEATURE(leak_sanitizer)
#define TURBO_HAVE_LEAK_SANITIZER 1
#elif defined(TURBO_HAVE_ADDRESS_SANITIZER)
// GCC or Clang using the LeakSanitizer integrated into AddressSanitizer.
#define TURBO_HAVE_LEAK_SANITIZER 1
#endif


// TURBO_HAVE_THREAD_LOCAL
//
// Checks whether C++11's `thread_local` storage duration specifier is
// supported.
#ifdef TURBO_HAVE_THREAD_LOCAL
#error TURBO_HAVE_THREAD_LOCAL cannot be directly set
#elif defined(__APPLE__)
// Notes:
// * Xcode's clang did not support `thread_local` until version 8, and
//   even then not for all iOS < 9.0.
// * Xcode 9.3 started disallowing `thread_local` for 32-bit iOS simulator
//   targeting iOS 9.x.
// * Xcode 10 moves the deployment target check for iOS < 9.0 to link time
//   making TURBO_HAVE_FEATURE unreliable there.
//
#if TURBO_HAVE_FEATURE(cxx_thread_local) && \
    !(TARGET_OS_IPHONE && __IPHONE_OS_VERSION_MIN_REQUIRED < __IPHONE_9_0)
#define TURBO_HAVE_THREAD_LOCAL 1
#endif
#else  // !defined(__APPLE__)
#define TURBO_HAVE_THREAD_LOCAL 1
#endif

// TURBO_ARRAY_SIZE()
//
// Returns the number of elements in an array as a compile-time constant, which
// can be used in defining new arrays. If you use this macro on a pointer by
// mistake, you will get a compile-time error.
#define TURBO_ARRAY_SIZE(array) \
  (sizeof(::turbo::macros_internal::ArraySizeHelper(array)))

namespace turbo {

namespace macros_internal {
// Note: this internal template function declaration is used by TURBO_ARRAY_SIZE.
// The function doesn't need a definition, as we only use its type.
template <typename T, size_t N>
auto ArraySizeHelper(const T (&array)[N]) -> char (&)[N];
}  // namespace macros_internal

}  // namespace turbo

#endif  // TURBO_PALTFORM_CONFIG_H_
