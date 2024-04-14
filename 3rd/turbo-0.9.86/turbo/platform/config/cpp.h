// Copyright 2023 The Elastic-AI Authors.
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


#ifndef TURBO_PLATFORM_CONFIG_CPP_H_
#define TURBO_PLATFORM_CONFIG_CPP_H_

#ifdef _MSVC_LANG
#  define TURBO_CPLUSPLUS _MSVC_LANG
#else
#  define TURBO_CPLUSPLUS __cplusplus
#endif

#define TURBO_CPLUSPLUS_98 199711L
#define TURBO_CPLUSPLUS_11 201103L
#define TURBO_CPLUSPLUS_14 201402L
#define TURBO_CPLUSPLUS_17 201703L
#define TURBO_CPLUSPLUS_17_9 201709L
#define TURBO_CPLUSPLUS_20 202002L

#if ((TURBO_CPLUSPLUS >= TURBO_CPLUSPLUS_20) &&                            \
     (!defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE > 9)) || \
    (TURBO_CPLUSPLUS >= TURBO_CPLUSPLUS_17_9 && TURBO_GCC_VERSION >= 1002)
#  define TURBO_CONSTEXPR20 constexpr
#else
#  define TURBO_CONSTEXPR20
#endif

#ifndef TURBO_CONSTEVAL
#  if ((TURBO_GCC_VERSION >= 1000 || TURBO_CLANG_VERSION >= 1101) && \
       (!defined(__apple_build_version__) ||                     \
        __apple_build_version__ >= 14000029L) &&                 \
       TURBO_CPLUSPLUS >= 202002L) ||                              \
      (defined(__cpp_consteval) &&                               \
       (!TURBO_MSC_VERSION || _MSC_FULL_VER >= 193030704))
// consteval is broken in MSVC before VS2022 and Apple clang before 14.
#    define TURBO_CONSTEVAL consteval
#    define TURBO_HAS_CONSTEVAL
#  else
#    define TURBO_CONSTEVAL
#  endif
#endif

#ifndef TURBO_USE_NONTYPE_TEMPLATE_ARGS
#  if defined(__cpp_nontype_template_args) && \
      ((TURBO_GCC_VERSION >= 903 && TURBO_CPLUSPLUS >= 201709L) || \
       __cpp_nontype_template_args >= 201911L) && \
      !defined(__NVCOMPILER) && !defined(__LCC__)
#    define TURBO_USE_NONTYPE_TEMPLATE_ARGS 1
#  else
#    define TURBO_USE_NONTYPE_TEMPLATE_ARGS 0
#  endif
#endif

#endif  // TURBO_PLATFORM_CONFIG_CPP_H_
