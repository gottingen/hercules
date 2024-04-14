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

#ifndef TURBO_PLATFORM_CONFIG_CONFIG_HAVE_H_
#define TURBO_PLATFORM_CONFIG_CONFIG_HAVE_H_


// TURBO_HAVE_EXCEPTIONS
//
// Checks whether the compiler both supports and enables exceptions. Many
// compilers support a "no exceptions" mode that disables exceptions.
//
// Generally, when TURBO_HAVE_EXCEPTIONS is not defined:
//
// * Code using `throw` and `try` may not compile.
// * The `noexcept` specifier will still compile and behave as normal.
// * The `noexcept` operator may still return `false`.
//
// For further details, consult the compiler's documentation.
#ifdef TURBO_HAVE_EXCEPTIONS
#error TURBO_HAVE_EXCEPTIONS cannot be directly set.
#elif TURBO_HAVE_MIN_CLANG_VERSION(3, 6)
// Clang >= 3.6
#if TURBO_HAVE_FEATURE(cxx_exceptions)
#define TURBO_HAVE_EXCEPTIONS 1
#endif  // TURBO_HAVE_FEATURE(cxx_exceptions)
#elif defined(__clang__)
// Clang < 3.6
// http://releases.llvm.org/3.6.0/tools/clang/docs/ReleaseNotes.html#the-exceptions-macro
#if defined(__EXCEPTIONS) && TURBO_HAVE_FEATURE(cxx_exceptions)
#define TURBO_HAVE_EXCEPTIONS 1
#endif  // defined(__EXCEPTIONS) && TURBO_HAVE_FEATURE(cxx_exceptions)
// Handle remaining special cases and default to exceptions being supported.
#elif !(defined(__GNUC__) && (__GNUC__ < 5) && !defined(__EXCEPTIONS)) && \
    !(TURBO_HAVE_MIN_GNUC_VERSION(5, 0) &&                        \
      !defined(__cpp_exceptions)) &&                                      \
    !(defined(_MSC_VER) && !defined(_CPPUNWIND))
#define TURBO_HAVE_EXCEPTIONS 1
#endif



// TURBO_HAVE_TLS is defined to 1 when __thread should be supported.
// We assume __thread is supported on Linux or Asylo when compiled with Clang or
// compiled against libstdc++ with _GLIBCXX_HAVE_TLS defined.
#ifdef TURBO_HAVE_TLS
#error TURBO_HAVE_TLS cannot be directly set
#elif (defined(__linux__) || defined(__ASYLO__)) && \
    (defined(__clang__) || defined(_GLIBCXX_HAVE_TLS))
#define TURBO_HAVE_TLS 1
#endif

// TURBO_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE
//
// Checks whether `std::is_trivially_destructible<T>` is supported.
//
// Notes: All supported compilers using libc++ support this feature, as does
// gcc >= 4.8.1 using libstdc++, and Visual Studio.
#ifdef TURBO_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE
#error TURBO_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE cannot be directly set
#elif defined(_LIBCPP_VERSION) || defined(_MSC_VER) || \
    (defined(__clang__) && __clang_major__ >= 15) ||   \
    (!defined(__clang__) && defined(__GLIBCXX__) &&    \
     TURBO_HAVE_MIN_GNUC_VERSION(4, 8))
#define TURBO_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE 1
#endif

// TURBO_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE
//
// Checks whether `std::is_trivially_default_constructible<T>` and
// `std::is_trivially_copy_constructible<T>` are supported.

// TURBO_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE
//
// Checks whether `std::is_trivially_copy_assignable<T>` is supported.

// Notes: Clang with libc++ supports these features, as does gcc >= 7.4 with
// libstdc++, or gcc >= 8.2 with libc++, and Visual Studio (but not NVCC).
#if defined(TURBO_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE)
#error TURBO_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE cannot be directly set
#elif defined(TURBO_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE)
#error TURBO_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE cannot directly set
#elif (defined(__clang__) && defined(_LIBCPP_VERSION)) ||                    \
    (defined(__clang__) && __clang_major__ >= 15) ||                         \
    (!defined(__clang__) &&                                                  \
     ((TURBO_HAVE_MIN_GNUC_VERSION(7, 4) && defined(__GLIBCXX__)) || \
      (TURBO_HAVE_MIN_GNUC_VERSION(8, 2) &&                          \
       defined(_LIBCPP_VERSION)))) ||                                        \
    (defined(_MSC_VER) && !defined(__NVCC__) && !defined(__clang__))
#define TURBO_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE 1
#define TURBO_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE 1
#endif

// TURBO_HAVE_STD_IS_TRIVIALLY_COPYABLE
//
// Checks whether `std::is_trivially_copyable<T>` is supported.
//
// Notes: Clang 15+ with libc++ supports these features, GCC hasn't been tested.
#if defined(TURBO_HAVE_STD_IS_TRIVIALLY_COPYABLE)
#error TURBO_HAVE_STD_IS_TRIVIALLY_COPYABLE cannot be directly set
#elif defined(__clang__) && (__clang_major__ >= 15)
#define TURBO_HAVE_STD_IS_TRIVIALLY_COPYABLE 1
#endif

// TURBO_HAVE_INTRINSIC_INT128
//
// Checks whether the __int128 compiler extension for a 128-bit integral type is
// supported.
//
// Note: __SIZEOF_INT128__ is defined by Clang and GCC when __int128 is
// supported, but we avoid using it in certain cases:
// * On Clang:
//   * Building using Clang for Windows, where the Clang runtime library has
//     128-bit support only on LP64 architectures, but Windows is LLP64.
// * On Nvidia's nvcc:
//   * nvcc also defines __GNUC__ and __SIZEOF_INT128__, but not all versions
//     actually support __int128.
#ifdef TURBO_HAVE_INTRINSIC_INT128
#error TURBO_HAVE_INTRINSIC_INT128 cannot be directly set
#elif defined(__SIZEOF_INT128__)
#if (defined(__clang__) && !defined(_WIN32)) || \
    (defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 9) ||                \
    (defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__))
#define TURBO_HAVE_INTRINSIC_INT128 1
#elif defined(__CUDACC__)
// __CUDACC_VER__ is a full version number before CUDA 9, and is defined to a
// string explaining that it has been removed starting with CUDA 9. We use
// nested #ifs because there is no short-circuiting in the preprocessor.
// NOTE: `__CUDACC__` could be undefined while `__CUDACC_VER__` is defined.
#if __CUDACC_VER__ >= 70000
#define TURBO_HAVE_INTRINSIC_INT128 1
#endif  // __CUDACC_VER__ >= 70000
#endif  // defined(__CUDACC__)
#endif  // TURBO_HAVE_INTRINSIC_INT128


// macOS < 10.13 and iOS < 11 don't let you use <any>, <optional>, or <variant>
// even though the headers exist and are publicly noted to work, because the
// libc++ shared library shipped on the system doesn't have the requisite
//
// libc++ spells out the availability requirements in the file
// llvm-project/libcxx/include/__config via the #define
// _LIBCPP_AVAILABILITY_BAD_OPTIONAL_ACCESS.
//
// Unfortunately, Apple initially mis-stated the requirements as macOS < 10.14
// and iOS < 12 in the libc++ headers. This was corrected by
// https://github.com/llvm/llvm-project/commit/7fb40e1569dd66292b647f4501b85517e9247953
// which subsequently made it into the XCode 12.5 release. We need to match the
// old (incorrect) conditions when built with old XCode, but can use the
// corrected earlier versions with new XCode.
#if defined(__APPLE__) && defined(_LIBCPP_VERSION) &&               \
    ((_LIBCPP_VERSION >= 11000 && /* XCode 12.5 or later: */        \
      ((defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) &&   \
        __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101300) ||  \
       (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) &&  \
        __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 110000) || \
       (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) &&   \
        __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 40000) ||   \
       (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) &&      \
        __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 110000))) ||   \
     (_LIBCPP_VERSION < 11000 && /* Pre-XCode 12.5: */              \
      ((defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) &&   \
        __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101400) ||  \
       (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) &&  \
        __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 120000) || \
       (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) &&   \
        __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 50000) ||   \
       (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) &&      \
        __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 120000))))
#define TURBO_INTERNAL_APPLE_CXX17_TYPES_UNAVAILABLE 1
#else
#define TURBO_INTERNAL_APPLE_CXX17_TYPES_UNAVAILABLE 0
#endif

// TURBO_HAVE_MMAP
//
// Checks whether the platform has an mmap(2) implementation as defined in
// POSIX.1-2001.
#ifdef TURBO_HAVE_MMAP
#error TURBO_HAVE_MMAP cannot be directly set
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || \
    defined(_AIX) || defined(__ros__) || defined(__native_client__) ||    \
    defined(__asmjs__) || defined(__wasm__) || defined(__Fuchsia__) ||    \
    defined(__sun) || defined(__ASYLO__) || defined(__myriad2__) ||       \
    defined(__HAIKU__) || defined(__OpenBSD__) || defined(__NetBSD__) ||  \
    defined(__QNX__)
#define TURBO_HAVE_MMAP 1
#endif

// TURBO_HAVE_PTHREAD_GETSCHEDPARAM
//
// Checks whether the platform implements the pthread_(get|set)schedparam(3)
// functions as defined in POSIX.1-2001.
#ifdef TURBO_HAVE_PTHREAD_GETSCHEDPARAM
#error TURBO_HAVE_PTHREAD_GETSCHEDPARAM cannot be directly set
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || \
    defined(_AIX) || defined(__ros__) || defined(__OpenBSD__) ||          \
    defined(__NetBSD__)
#define TURBO_HAVE_PTHREAD_GETSCHEDPARAM 1
#endif

// TURBO_HAVE_SCHED_GETCPU
//
// Checks whether sched_getcpu is available.
#ifdef TURBO_HAVE_SCHED_GETCPU
#error TURBO_HAVE_SCHED_GETCPU cannot be directly set
#elif defined(__linux__)
#define TURBO_HAVE_SCHED_GETCPU 1
#endif

// TURBO_HAVE_SCHED_YIELD
//
// Checks whether the platform implements sched_yield(2) as defined in
// POSIX.1-2001.
#ifdef TURBO_HAVE_SCHED_YIELD
#error TURBO_HAVE_SCHED_YIELD cannot be directly set
#elif defined(__linux__) || defined(__ros__) || defined(__native_client__)
#define TURBO_HAVE_SCHED_YIELD 1
#endif


// TURBO_HAVE_ALARM
//
// Checks whether the platform supports the <signal.h> header and alarm(2)
// function as standardized in POSIX.1-2001.
#ifdef TURBO_HAVE_ALARM
#error TURBO_HAVE_ALARM cannot be directly set
#elif defined(__GOOGLE_GRTE_VERSION__)
// feature tests for Google's GRTE
#define TURBO_HAVE_ALARM 1
#elif defined(__GLIBC__)
// feature test for glibc
#define TURBO_HAVE_ALARM 1
#elif defined(_MSC_VER)
// feature tests for Microsoft's library
#elif defined(__MINGW32__)
// mingw32 doesn't provide alarm(2):
// https://osdn.net/projects/mingw/scm/git/mingw-org-wsl/blobs/5.2-trunk/mingwrt/include/unistd.h
// mingw-w64 provides a no-op implementation:
// https://sourceforge.net/p/mingw-w64/mingw-w64/ci/master/tree/mingw-w64-crt/misc/alarm.c
#elif defined(__EMSCRIPTEN__)
// emscripten doesn't support signals
#elif defined(__Fuchsia__)
// Signals don't exist on fuchsia.
#elif defined(__native_client__)
#else
// other standard libraries
#define TURBO_HAVE_ALARM 1
#endif


// `TURBO_COMPILER_HAVE_RTTI` determines whether turbo is being compiled with
// RTTI support.
#ifdef TURBO_COMPILER_HAVE_RTTI
#error TURBO_COMPILER_HAVE_RTTI cannot be directly set
#elif !defined(__GNUC__) || defined(__GXX_RTTI)
#define TURBO_COMPILER_HAVE_RTTI 1
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)

// TURBO_HAVE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
//
// Class template argument deduction is a language feature added in C++17.
#ifdef TURBO_HAVE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
#error "TURBO_HAVE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION cannot be directly set."
#elif defined(__cpp_deduction_guides)
#define TURBO_HAVE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION 1
#endif

// TURBO_HAVE_CONSTANT_EVALUATED is used for compile-time detection of
// constant evaluation support through `turbo::is_constant_evaluated`.
#ifdef TURBO_HAVE_CONSTANT_EVALUATED
#error TURBO_HAVE_CONSTANT_EVALUATED cannot be directly set
#endif
#ifdef __cpp_lib_is_constant_evaluated
#define TURBO_HAVE_CONSTANT_EVALUATED 1
#elif TURBO_HAVE_BUILTIN(__builtin_is_constant_evaluated)
#define TURBO_HAVE_CONSTANT_EVALUATED 1
#endif

#endif  // TURBO_PLATFORM_CONFIG_CONFIG_HAVE_H_
