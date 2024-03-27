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
//

#ifndef COLLIE_BASE_MACROS_H_
#define COLLIE_BASE_MACROS_H_

#ifndef COLLIE_LIKELY
#if defined(__GNUC__)
#define COLLIE_LIKELY(x) (__builtin_expect((x), 1))
#define COLLIE_UNLIKELY(x) (__builtin_expect((x), 0))
#else
#define COLLIE_LIKELY(x) (x)
#define COLLIE_UNLIKELY(x) (x)
#endif
#endif

#ifdef __has_builtin
#  define COLLIE_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define COLLIE_HAS_BUILTIN(x) 0
#endif

/// COLLIE_BUILTIN_UNREACHABLE - On compilers which support it, expands
/// to an expression which states that it is undefined behavior for the
/// compiler to reach this point.  Otherwise is not defined.
///
/// '#else' is intentionally left out so that other macro logic (e.g.,
/// COLLIE_ASSUME_ALIGNED and llvm_unreachable()) can detect whether
/// COLLIE_BUILTIN_UNREACHABLE has a definition.
#if COLLIE_HAS_BUILTIN(__builtin_unreachable) || defined(__GNUC__)
# define COLLIE_BUILTIN_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
# define COLLIE_BUILTIN_UNREACHABLE __assume(false)
#endif

/// \macro COLLIE_ASSUME_ALIGNED
/// Returns a pointer with an assumed alignment.
#if COLLIE_HAS_BUILTIN(__builtin_assume_aligned) || defined(__GNUC__)
# define COLLIE_ASSUME_ALIGNED(p, a) __builtin_assume_aligned(p, a)
#elif defined(COLLIE_BUILTIN_UNREACHABLE)
# define COLLIE_ASSUME_ALIGNED(p, a) \
           (((uintptr_t(p) % (a)) == 0) ? (p) : (COLLIE_BUILTIN_UNREACHABLE, (p)))
#else
# define COLLIE_ASSUME_ALIGNED(p, a) (p)
#endif

#ifdef __has_attribute
#define COLLIE_HAS_ATTR(attr) __has_attribute(attr)
#else
#define COLLIE_HAS_ATTR(attr) 0
#endif
#if defined(_MSC_VER)
#define COLLIE_ATTRIBUTE_RETURNS_NONNULL _Ret_notnull_
#elif COLLIE_HAS_ATTR(returns_nonnull)
#define COLLIE_ATTRIBUTE_RETURNS_NONNULL __attribute__((returns_nonnull))
#else
#define COLLIE_ATTRIBUTE_RETURNS_NONNULL
#endif


#endif  // COLLIE_BASE_MACROS_H_
