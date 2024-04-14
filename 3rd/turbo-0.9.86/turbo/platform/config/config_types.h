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

#ifndef TURBO_PLATFORM_CONFIG_CONFIG_TYPES_H_
#define TURBO_PLATFORM_CONFIG_CONFIG_TYPES_H_

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cfloat>
#include <cinttypes>
#include "turbo/platform/config/compiler.h"
#include "turbo/platform/config/compiler_traits.h"
#include "turbo/platform/config/platform.h"


// ------------------------------------------------------------------------
// bool8_t
// The definition of a bool8_t is controversial with some, as it doesn't
// act just like built-in bool. For example, you can assign -100 to it.
//
#ifndef BOOL8_T_DEFINED // If the user hasn't already defined this...
#define BOOL8_T_DEFINED
#if defined(TURBO_COMPILER_MSVC) || (defined(TURBO_COMPILER_INTEL) && defined(TURBO_PLATFORM_WINDOWS))
#if defined(__cplusplus)
typedef bool bool8_t;
#else
typedef int8_t bool8_t;
#endif
#else // TURBO_COMPILER_GNUC generally uses 4 bytes per bool.
typedef int8_t bool8_t;
#endif
#endif

// ------------------------------------------------------------------------
// Character types
//
#if defined(TURBO_COMPILER_MSVC)
#if defined(TURBO_WCHAR_T_NON_NATIVE)
// In this case, wchar_t is not defined unless we include
// wchar.h or if the compiler makes it built-in.
#ifdef TURBO_COMPILER_MSVC
#pragma warning(push, 3)
#endif
#include <wchar.h>
#ifdef TURBO_COMPILER_MSVC
#pragma warning(pop)
#endif
#endif
#endif

// ------------------------------------------------------------------------
// char8_t  -- Guaranteed to be equal to the compiler's char data type.
//             Some compilers implement char8_t as unsigned, though char
//             is usually set to be signed.
//
// char16_t -- This is set to be an unsigned 16 bit value. If the compiler
//             has wchar_t as an unsigned 16 bit value, then char16_t is
//             set to be the same thing as wchar_t in order to allow the
//             user to use char16_t with standard wchar_t functions.
//
// char32_t -- This is set to be an unsigned 32 bit value. If the compiler
//             has wchar_t as an unsigned 32 bit value, then char32_t is
//             set to be the same thing as wchar_t in order to allow the
//             user to use char32_t with standard wchar_t functions.
//
// TURBO_CHAR8_UNIQUE
// TURBO_CHAR16_NATIVE
// TURBO_CHAR32_NATIVE
// TURBO_WCHAR_UNIQUE
//
// VS2010 unilaterally defines char16_t and char32_t in its yvals.h header
// unless _HAS_CHAR16_T_LANGUAGE_SUPPORT or _CHAR16T are defined.
// However, VS2010 does not support the C++0x u"" and U"" string literals,
// which makes its definition of char16_t and char32_t somewhat useless.
// Until VC++ supports string literals, the build system should define
// _CHAR16T and let TBBase define char16_t and TURBO_CHAR16.
//
// GCC defines char16_t and char32_t in the C compiler in -std=gnu99 mode,
// as __CHAR16_TYPE__ and __CHAR32_TYPE__, and for the C++ compiler
// in -std=c++0x and -std=gnu++0x modes, as char16_t and char32_t too.
//
// The TURBO_WCHAR_UNIQUE symbol is defined to 1 if wchar_t is distinct from
// char8_t, char16_t, and char32_t, and defined to 0 if not. In some cases,
// if the compiler does not support char16_t/char32_t, one of these two types
// is typically a typedef or define of wchar_t. For compilers that support
// the C++11 unicode character types often overloads must be provided to
// support existing code that passes a wide char string to a function that
// takes a unicode string.
//
// The TURBO_CHAR8_UNIQUE symbol is defined to 1 if char8_t is distinct type
// from char in the type system, and defined to 0 if otherwise.

#if !defined(TURBO_CHAR16_NATIVE)
// To do: Change this to be based on TURBO_COMPILER_NO_NEW_CHARACTER_TYPES.
#if defined(_MSC_VER) && (_MSC_VER >= 1600) && defined(_HAS_CHAR16_T_LANGUAGE_SUPPORT) && _HAS_CHAR16_T_LANGUAGE_SUPPORT // VS2010+
#define TURBO_CHAR16_NATIVE 1
#elif defined(TURBO_COMPILER_CLANG) && defined(TURBO_COMPILER_CPP11_ENABLED)
#if __has_feature(cxx_unicode_literals)
#define TURBO_CHAR16_NATIVE 1
#elif (TURBO_COMPILER_VERSION >= 300) && !(defined(TURBO_PLATFORM_IPHONE) || defined(TURBO_PLATFORM_OSX))
#define TURBO_CHAR16_NATIVE 1
#elif defined(TURBO_PLATFORM_APPLE)
#define TURBO_CHAR16_NATIVE 1
#else
#define TURBO_CHAR16_NATIVE 0
#endif
#elif defined(__EDG_VERSION__) && (__EDG_VERSION__ >= 404) && defined(__CHAR16_TYPE__) && defined(TURBO_COMPILER_CPP11_ENABLED)// EDG 4.4+.
#define TURBO_CHAR16_NATIVE 1
#elif defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION >= 4004) && !defined(TURBO_COMPILER_EDG) && (defined(TURBO_COMPILER_CPP11_ENABLED) || defined(__STDC_VERSION__)) // g++ (C++ compiler) 4.4+ with -std=c++0x or gcc (C compiler) 4.4+ with -std=gnu99
#define TURBO_CHAR16_NATIVE 1
#else
#define TURBO_CHAR16_NATIVE 0
#endif
#endif

#if !defined(TURBO_CHAR32_NATIVE)                    // Microsoft currently ties char32_t language support to char16_t language support. So we use CHAR16_T here.
// To do: Change this to be based on TURBO_COMPILER_NO_NEW_CHARACTER_TYPES.
#if defined(_MSC_VER) && (_MSC_VER >= 1600) && defined(_HAS_CHAR16_T_LANGUAGE_SUPPORT) && _HAS_CHAR16_T_LANGUAGE_SUPPORT // VS2010+
#define TURBO_CHAR32_NATIVE 1
#elif defined(TURBO_COMPILER_CLANG) && defined(TURBO_COMPILER_CPP11_ENABLED)
#if __has_feature(cxx_unicode_literals)
#define TURBO_CHAR32_NATIVE 1
#elif (TURBO_COMPILER_VERSION >= 300) && !(defined(TURBO_PLATFORM_IPHONE) || defined(TURBO_PLATFORM_OSX))
#define TURBO_CHAR32_NATIVE 1
#elif defined(TURBO_PLATFORM_APPLE)
#define TURBO_CHAR32_NATIVE 1
#else
#define TURBO_CHAR32_NATIVE 0
#endif
#elif defined(__EDG_VERSION__) && (__EDG_VERSION__ >= 404) && defined(__CHAR32_TYPE__) && defined(TURBO_COMPILER_CPP11_ENABLED)// EDG 4.4+.
#define TURBO_CHAR32_NATIVE 1
#elif defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION >= 4004) && !defined(TURBO_COMPILER_EDG) && (defined(TURBO_COMPILER_CPP11_ENABLED) || defined(__STDC_VERSION__)) // g++ (C++ compiler) 4.4+ with -std=c++0x or gcc (C compiler) 4.4+ with -std=gnu99
#define TURBO_CHAR32_NATIVE 1
#else
#define TURBO_CHAR32_NATIVE 0
#endif
#endif


#if TURBO_CHAR16_NATIVE || TURBO_CHAR32_NATIVE
#define TURBO_WCHAR_UNIQUE 1
#else
#define TURBO_WCHAR_UNIQUE 0
#endif

// TURBO_CHAR8_UNIQUE
//
// Check for char8_t support in the cpp type system. Moving forward from c++20,
// the char8_t type allows users to overload function for character encoding.
//
// TURBO_CHAR8_UNIQUE is 1 when the type is a unique in the type system and
// can there be used as a valid overload. TURBO_CHAR8_UNIQUE is 0 otherwise.
//
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0482r6.html
//
#ifdef __cpp_char8_t
#define CHAR8_T_DEFINED
#define TURBO_CHAR8_UNIQUE 1
#else
#define TURBO_CHAR8_UNIQUE 0
#endif




#ifndef CHAR8_T_DEFINED // If the user hasn't already defined these...
#define CHAR8_T_DEFINED
//using char8_t = unsigned char;
/*
#if defined(TURBO_PLATFORM_APPLE)
        #define char8_t char    // The Apple debugger is too stupid to realize char8_t is typedef'd to char, so we #define it.
#else
        typedef char char8_t;
#endif
 */

#if TURBO_CHAR16_NATIVE
// In C++, char16_t and char32_t are already defined by the compiler.
// In MS C, char16_t and char32_t are already defined by the compiler/standard library.
// In GCC C, __CHAR16_TYPE__ and __CHAR32_TYPE__ are defined instead, and we must define char16_t and char32_t from these.
#if defined(__GNUC__) && !defined(__GXX_EXPERIMENTAL_CXX0X__) && defined(__CHAR16_TYPE__) // If using GCC and compiling in C...
typedef __CHAR16_TYPE__ char16_t;
typedef __CHAR32_TYPE__ char32_t;
#endif
#elif (TURBO_WCHAR_SIZE == 2)
#if (defined(_MSC_VER) && (_MSC_VER >= 1600)) // if VS2010+ or using platforms that use Dinkumware under a compiler that doesn't natively support C++11 char16_t.
#if !defined(_CHAR16T)
#define _CHAR16T
#endif
#if !defined(_HAS_CHAR16_T_LANGUAGE_SUPPORT) || !_HAS_CHAR16_T_LANGUAGE_SUPPORT
typedef wchar_t  char16_t;
typedef uint32_t char32_t;
#endif
#else
typedef wchar_t  char16_t;
typedef uint32_t char32_t;
#endif
#else
typedef uint16_t char16_t;
#if defined(__cplusplus)
typedef wchar_t  char32_t;
#else
typedef uint32_t char32_t;
#endif
#endif
#endif


// CHAR8_MIN, CHAR8_MAX, etc.
//
#define TURBO_LIMITS_DIGITS_S(T)  ((sizeof(T) * 8) - 1)
#define TURBO_LIMITS_DIGITS_U(T)  ((sizeof(T) * 8))
#define TURBO_LIMITS_DIGITS(T)    ((TURBO_LIMITS_IS_SIGNED(T) ? TURBO_LIMITS_DIGITS_S(T) : TURBO_LIMITS_DIGITS_U(T)))
#define TURBO_LIMITS_IS_SIGNED(T) ((T)(-1) < 0)
#define TURBO_LIMITS_MIN_S(T)     ((T)((T)1 << TURBO_LIMITS_DIGITS_S(T)))
#define TURBO_LIMITS_MIN_U(T)     ((T)0)
#define TURBO_LIMITS_MIN(T)       ((TURBO_LIMITS_IS_SIGNED(T) ? TURBO_LIMITS_MIN_S(T) : TURBO_LIMITS_MIN_U(T)))
#define TURBO_LIMITS_MAX_S(T)     ((T)(((((T)1 << (TURBO_LIMITS_DIGITS(T) - 1)) - 1) << 1) + 1))
#define TURBO_LIMITS_MAX_U(T)     ((T)~(T)0)
#define TURBO_LIMITS_MAX(T)       ((TURBO_LIMITS_IS_SIGNED(T) ? TURBO_LIMITS_MAX_S(T) : TURBO_LIMITS_MAX_U(T)))

#if !defined(CHAR8_MIN)
#define CHAR8_MIN TURBO_LIMITS_MIN(char8_t)
#endif

#if !defined(CHAR8_MAX)
#define CHAR8_MAX TURBO_LIMITS_MAX(char8_t)
#endif

#if !defined(CHAR16_MIN)
#define CHAR16_MIN TURBO_LIMITS_MIN(char16_t)
#endif

#if !defined(CHAR16_MAX)
#define CHAR16_MAX TURBO_LIMITS_MAX(char16_t)
#endif

#if !defined(CHAR32_MIN)
#define CHAR32_MIN TURBO_LIMITS_MIN(char32_t)
#endif

#if !defined(CHAR32_MAX)
#define CHAR32_MAX TURBO_LIMITS_MAX(char32_t)
#endif




// TURBO_CHAR8 / TURBO_CHAR16 / TURBO_CHAR32 / TURBO_WCHAR
//
// Supports usage of portable string constants.
//
// Example usage:
//     const char16_t* str = TURBO_CHAR16("Hello world");
//     const char32_t* str = TURBO_CHAR32("Hello world");
//     const char16_t  c   = TURBO_CHAR16('\x3001');
//     const char32_t  c   = TURBO_CHAR32('\x3001');
//
#ifndef TURBO_CHAR8
#if TURBO_CHAR8_UNIQUE
#define TURBO_CHAR8(s) u8 ## s
#else
#define TURBO_CHAR8(s) s
#endif
#endif

#ifndef TURBO_WCHAR
#define TURBO_WCHAR_(s) L ## s
#define TURBO_WCHAR(s)  TURBO_WCHAR_(s)
#endif

#ifndef TURBO_CHAR16
#if TURBO_CHAR16_NATIVE && !defined(_MSC_VER) // Microsoft doesn't support char16_t string literals.
#define TURBO_CHAR16_(s) u ## s
#define TURBO_CHAR16(s)  TURBO_CHAR16_(s)
#elif (TURBO_WCHAR_SIZE == 2)
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && defined(__cplusplus) // VS2015 supports u"" string literals.
#define TURBO_CHAR16_(s) u ## s
#define TURBO_CHAR16(s)  TURBO_CHAR16_(s)
#else
#define TURBO_CHAR16_(s) L ## s
#define TURBO_CHAR16(s)  TURBO_CHAR16_(s)
#endif
#else
//#define TURBO_CHAR16(s) // Impossible to implement efficiently.
#endif
#endif

#ifndef TURBO_CHAR32
#if TURBO_CHAR32_NATIVE && !defined(_MSC_VER) // Microsoft doesn't support char32_t string literals.
#define TURBO_CHAR32_(s) U ## s
#define TURBO_CHAR32(s)  TURBO_CHAR32_(s)
#elif (TURBO_WCHAR_SIZE == 2)
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && defined(__cplusplus) // VS2015 supports u"" string literals.
#define TURBO_CHAR32_(s) U ## s
#define TURBO_CHAR32(s)  TURBO_CHAR32_(s)
#else
//#define TURBO_CHAR32(s) // Impossible to implement.
#endif
#elif (TURBO_WCHAR_SIZE == 4)
#define TURBO_CHAR32_(s) L ## s
#define TURBO_CHAR32(s)  TURBO_CHAR32_(s)
#else
#error Unexpected size of wchar_t
#endif
#endif

#endif // TURBO_PLATFORM_CONFIG_CONFIG_TYPES_H_
