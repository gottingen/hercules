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

#ifndef TURBO_PLATFORM_CONFIG_WCHAR_CONFIG_H_
#define TURBO_PLATFORM_CONFIG_WCHAR_CONFIG_H_

#include "turbo/platform/config/compiler.h"
#include "turbo/platform/config/platform.h"
// ------------------------------------------------------------------------
// wchar_t
// Here we define:
//    TURBO_WCHAR_T_NON_NATIVE
//    TURBO_WCHAR_SIZE = <sizeof(wchar_t)>
//
#ifndef TURBO_WCHAR_T_NON_NATIVE
// Compilers that always implement wchar_t as native include:
//     COMEAU, new SN, and other EDG-based compilers.
//     GCC
//     Borland
//     SunPro
//     IBM Visual Age
#if defined(TURBO_COMPILER_INTEL)
#if (TURBO_COMPILER_VERSION < 700)
#define TURBO_WCHAR_T_NON_NATIVE 1
#else
#if (!defined(_WCHAR_T_DEFINED) && !defined(_WCHAR_T))
#define TURBO_WCHAR_T_NON_NATIVE 1
#endif
#endif
#elif defined(TURBO_COMPILER_MSVC) ||                                          \
    (defined(TURBO_COMPILER_CLANG) && defined(TURBO_PLATFORM_WINDOWS))
#ifndef _NATIVE_WCHAR_T_DEFINED
#define TURBO_WCHAR_T_NON_NATIVE 1
#endif
#elif defined(__EDG_VERSION__) &&                                              \
    (!defined(_WCHAR_T) &&                                                     \
     (__EDG_VERSION__ < 400)) // EDG prior to v4 uses _WCHAR_T to indicate if
                              // wchar_t is native. v4+ may define something
                              // else, but we're not currently aware of it.
#define TURBO_WCHAR_T_NON_NATIVE 1
#endif
#endif


#if defined(_MSC_VER)
// In very old versions of MSVC and when the /Zc:wchar_t flag is off, wchar_t is
// a typedef for unsigned short.  Otherwise wchar_t is mapped to the __wchar_t
// builtin type.  We need to make sure not to define operator wchar_t()
// alongside operator unsigned short() in these instances.
#define TURBO_WCHAR_T __wchar_t
#if defined(_M_X64) && !defined(_M_ARM64EC)
#include <intrin.h>
#pragma intrinsic(_umul128)
#endif  // defined(_M_X64)
#else   // defined(_MSC_VER)
#define TURBO_WCHAR_T wchar_t
#endif  // defined(_MSC_VER)


#ifndef TURBO_WCHAR_SIZE   // If the user hasn't specified that it is a given
                           // size...
#if defined(__WCHAR_MAX__) // GCC defines this for most platforms.
#if (__WCHAR_MAX__ == 2147483647) || (__WCHAR_MAX__ == 4294967295)
#define TURBO_WCHAR_SIZE 4
#elif (__WCHAR_MAX__ == 32767) || (__WCHAR_MAX__ == 65535)
#define TURBO_WCHAR_SIZE 2
#elif (__WCHAR_MAX__ == 127) || (__WCHAR_MAX__ == 255)
#define TURBO_WCHAR_SIZE 1
#else
#define TURBO_WCHAR_SIZE 4
#endif
#elif defined(WCHAR_MAX) // The SN and Arm compilers define this.
#if (WCHAR_MAX == 2147483647) || (WCHAR_MAX == 4294967295)
#define TURBO_WCHAR_SIZE 4
#elif (WCHAR_MAX == 32767) || (WCHAR_MAX == 65535)
#define TURBO_WCHAR_SIZE 2
#elif (WCHAR_MAX == 127) || (WCHAR_MAX == 255)
#define TURBO_WCHAR_SIZE 1
#else
#define TURBO_WCHAR_SIZE 4
#endif
#elif defined(                                                                 \
    __WCHAR_BIT) // Green Hills (and other versions of EDG?) uses this.
#if (__WCHAR_BIT == 16)
#define TURBO_WCHAR_SIZE 2
#elif (__WCHAR_BIT == 32)
#define TURBO_WCHAR_SIZE 4
#elif (__WCHAR_BIT == 8)
#define TURBO_WCHAR_SIZE 1
#else
#define TURBO_WCHAR_SIZE 4
#endif
#elif defined(_WCMAX) // The SN and Arm compilers define this.
#if (_WCMAX == 2147483647) || (_WCMAX == 4294967295)
#define TURBO_WCHAR_SIZE 4
#elif (_WCMAX == 32767) || (_WCMAX == 65535)
#define TURBO_WCHAR_SIZE 2
#elif (_WCMAX == 127) || (_WCMAX == 255)
#define TURBO_WCHAR_SIZE 1
#else
#define TURBO_WCHAR_SIZE 4
#endif
#elif defined(TURBO_PLATFORM_UNIX)
// It is standard on Unix to have wchar_t be int32_t or uint32_t.
// All versions of GNUC default to a 32 bit wchar_t, but EA has used
// the -fshort-wchar GCC command line option to force it to 16 bit.
// If you know that the compiler is set to use a wchar_t of other than
// the default, you need to manually define TURBO_WCHAR_SIZE for the build.
#define TURBO_WCHAR_SIZE 4
#else
// It is standard on Windows to have wchar_t be uint16_t.  GCC
// defines wchar_t as int by default.  Electronic Arts has
// standardized on wchar_t being an unsigned 16 bit value on all
// console platforms. Given that there is currently no known way to
// tell at preprocessor time what the size of wchar_t is, we declare
// it to be 2, as this is the Electronic Arts standard. If you have
// TURBO_WCHAR_SIZE != sizeof(wchar_t), then your code might not be
// broken, but it also won't work with wchar libraries and data from
// other parts of EA. Under GCC, you can force wchar_t to two bytes
// with the -fshort-wchar compiler argument.
#define TURBO_WCHAR_SIZE 2
#endif
#endif


// ------------------------------------------------------------------------
// The C++ standard defines size_t as a built-in type. Some compilers are
// not standards-compliant in this respect, so we need an additional include.
// The case is similar with wchar_t under C++.

#if defined(TURBO_COMPILER_GNUC) || defined(TURBO_COMPILER_MSVC) || defined(TURBO_WCHAR_T_NON_NATIVE) || defined(TURBO_PLATFORM_SONY)
#if defined(TURBO_COMPILER_MSVC)
#pragma warning(push, 0)
#pragma warning(disable: 4265 4365 4836 4574)
#endif
#include <stddef.h>
#if defined(TURBO_COMPILER_MSVC)
#pragma warning(pop)
#endif
#endif

#endif // TURBO_PLATFORM_CONFIG_WCHAR_CONFIG_H_
