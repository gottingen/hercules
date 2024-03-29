// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// The following version macro is very similar to the one in pybind11
#if !(defined(_MSC_VER) && __cplusplus == 199711L) && !defined(__INTEL_COMPILER)
#if __cplusplus >= 201402L
#define CLI11_CPP14
#if __cplusplus >= 201703L
#define CLI11_CPP17
#if __cplusplus > 201703L
#define CLI11_CPP20
#endif
#endif
#endif
#elif defined(_MSC_VER) && __cplusplus == 199711L
// MSVC sets _MSVC_LANG rather than __cplusplus (supposedly until the standard is fully implemented)
// Unless you use the /Zc:__cplusplus flag on Visual Studio 2017 15.7 Preview 3 or newer
#if _MSVC_LANG >= 201402L
#define CLI11_CPP14
#if _MSVC_LANG > 201402L && _MSC_VER >= 1910
#define CLI11_CPP17
#if _MSVC_LANG > 201703L && _MSC_VER >= 1910
#define CLI11_CPP20
#endif
#endif
#endif
#endif

/** detection of rtti */
#ifndef CLI11_USE_STATIC_RTTI
#if(defined(_HAS_STATIC_RTTI) && _HAS_STATIC_RTTI)
#define CLI11_USE_STATIC_RTTI 1
#elif defined(__cpp_rtti)
#if(defined(_CPPRTTI) && _CPPRTTI == 0)
#define CLI11_USE_STATIC_RTTI 1
#else
#define CLI11_USE_STATIC_RTTI 0
#endif
#elif(defined(__GCC_RTTI) && __GXX_RTTI)
#define CLI11_USE_STATIC_RTTI 0
#else
#define CLI11_USE_STATIC_RTTI 1
#endif
#endif


/** <codecvt> availability */
#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER) && __GNUC__ < 5
#define COLLIE_HAS_CODECVT 0
#else
#define COLLIE_HAS_CODECVT 1
#include <codecvt>
#endif

/** disable deprecations */
#if defined(__GNUC__)  // GCC or clang
#define CLI11_DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define CLI11_DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")

#define CLI11_DIAGNOSTIC_IGNORE_DEPRECATED _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")

#elif defined(_MSC_VER)
#define CLI11_DIAGNOSTIC_PUSH __pragma(warning(push))
#define CLI11_DIAGNOSTIC_POP __pragma(warning(pop))

#define CLI11_DIAGNOSTIC_IGNORE_DEPRECATED __pragma(warning(disable : 4996))

#else
#define CLI11_DIAGNOSTIC_PUSH
#define CLI11_DIAGNOSTIC_POP

#define CLI11_DIAGNOSTIC_IGNORE_DEPRECATED

#endif
