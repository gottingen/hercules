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

#ifndef TURBO_PLATFORM_CONFIG_ATTRIBUTE_VARIABLE_H_
#define TURBO_PLATFORM_CONFIG_ATTRIBUTE_VARIABLE_H_

#include "turbo/platform/config/compiler_traits.h"


// TURBO_ATTRIBUTE_LIFETIME_BOUND indicates that a resource owned by a function
// parameter or implicit object parameter is retained by the return value of the
// annotated function (or, for a parameter of a constructor, in the value of the
// constructed object). This attribute causes warnings to be produced if a
// temporary object does not live long enough.
//
// When applied to a reference parameter, the referenced object is assumed to be
// retained by the return value of the function. When applied to a non-reference
// parameter (for example, a pointer or a class type), all temporaries
// referenced by the parameter are assumed to be retained by the return value of
// the function.
//
// See also the upstream documentation:
// https://clang.llvm.org/docs/AttributeReference.html#lifetimebound
#if TURBO_HAVE_CPP_ATTRIBUTE(clang::lifetimebound)
#define TURBO_ATTRIBUTE_LIFETIME_BOUND [[clang::lifetimebound]]
#elif TURBO_HAVE_ATTRIBUTE(lifetimebound)
#define TURBO_ATTRIBUTE_LIFETIME_BOUND __attribute__((lifetimebound))
#else
#define TURBO_ATTRIBUTE_LIFETIME_BOUND
#endif



// TURBO_ATTRIBUTE_PACKED
//
// Instructs the compiler not to use natural alignment for a tagged data
// structure, but instead to reduce its alignment to 1.
//
// Therefore, DO NOT APPLY THIS ATTRIBUTE TO STRUCTS CONTAINING ATOMICS. Doing
// so can cause atomic variables to be mis-aligned and silently violate
// atomicity on x86.
//
// This attribute can either be applied to members of a structure or to a
// structure in its entirety. Applying this attribute (judiciously) to a
// structure in its entirety to optimize the memory footprint of very
// commonly-used structs is fine. Do not apply this attribute to a structure in
// its entirety if the purpose is to control the offsets of the members in the
// structure. Instead, apply this attribute only to structure members that need
// it.
//
// When applying TURBO_ATTRIBUTE_PACKED only to specific structure members the
// natural alignment of structure members not annotated is preserved. Aligned
// member accesses are faster than non-aligned member accesses even if the
// targeted microprocessor supports non-aligned accesses.
#if TURBO_HAVE_ATTRIBUTE(packed) || (defined(__GNUC__) && !defined(__clang__))
#define TURBO_ATTRIBUTE_PACKED __attribute__((__packed__))
#else
#define TURBO_ATTRIBUTE_PACKED
#endif

// ------------------------------------------------------------------------
// TURBO_RESTRICT
//
// The C99 standard defines a new keyword, restrict, which allows for the
// improvement of code generation regarding memory usage. Compilers can
// generate significantly faster code when you are able to use restrict.
//
// Example usage:
//    void DoSomething(char* TURBO_RESTRICT p1, char* TURBO_RESTRICT p2);
//
#ifndef TURBO_RESTRICT
#if defined(TURBO_COMPILER_MSVC) && (TURBO_COMPILER_VERSION >= 1400) // If VC8 (VS2005) or later...
#define TURBO_RESTRICT __restrict
#elif defined(TURBO_COMPILER_CLANG)
#define TURBO_RESTRICT __restrict
#elif defined(TURBO_COMPILER_GNUC)     // Includes GCC and other compilers emulating GCC.
#define TURBO_RESTRICT __restrict  // GCC defines 'restrict' (as opposed to __restrict) in C99 mode only.
#elif defined(TURBO_COMPILER_ARM)
#define TURBO_RESTRICT __restrict
#elif defined(TURBO_COMPILER_IS_C99)
#define TURBO_RESTRICT restrict
#else
// If the compiler didn't support restricted pointers, defining TURBO_RESTRICT
// away would result in compiling and running fine but you just wouldn't
// the same level of optimization. On the other hand, all the major compilers
// support restricted pointers.
#define TURBO_RESTRICT
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_UNUSED
//
// Makes compiler warnings about unused variables go away.
//
// Example usage:
//    void Function(int x)
//    {
//        int y;
//        TURBO_UNUSED(x);
//        TURBO_UNUSED(y);
//    }
//
#ifndef TURBO_UNUSED
// The EDG solution below is pretty weak and needs to be augmented or replaced.
// It can't handle the C language, is limited to places where template declarations
// can be used, and requires the type x to be usable as a functions reference argument.
#if defined(__cplusplus) && defined(__EDG__)
template <typename T>
inline void TBBaseUnused(T const volatile & x) { (void)x; }
#define TURBO_UNUSED(x) TBBaseUnused(x)
#else
#define TURBO_UNUSED(x) (void)x
#endif
#endif

// TURBO_MAY_ALIAS
//
// Defined as a macro that wraps the GCC may_alias attribute. This attribute
// has no significance for VC++ because VC++ doesn't support the concept of
// strict aliasing. Users should avoid writing code that breaks strict
// aliasing rules; TURBO_MAY_ALIAS is for cases with no alternative.
//
// Example usage:
//    void* TURBO_MAY_ALIAS gPtr = NULL;
//
// Example usage:
//    typedef void* TURBO_MAY_ALIAS pvoid_may_alias;
//    pvoid_may_alias gPtr = NULL;
//
#if TURBO_MAY_ALIAS_AVAILABLE
#define TURBO_MAY_ALIAS __attribute__((__may_alias__))
#else
#define TURBO_MAY_ALIAS
#endif

// ------------------------------------------------------------------------
// TURBO_LIKELY / TURBO_UNLIKELY
//
// Defined as a macro which gives a hint to the compiler for branch
// prediction. GCC gives you the ability to manually give a hint to
// the compiler about the result of a comparison, though it's often
// best to compile shipping code with profiling feedback under both
// GCC (-fprofile-arcs) and VC++ (/LTCG:PGO, etc.). However, there
// are times when you feel very sure that a boolean expression will
// usually evaluate to either true or false and can help the compiler
// by using an explicity directive...
//
// Example usage:
//    if(TURBO_LIKELY(a == 0)) // Tell the compiler that a will usually equal 0.
//       { ... }
//
// Example usage:
//    if(TURBO_UNLIKELY(a == 0)) // Tell the compiler that a will usually not equal 0.
//       { ... }
//
#ifndef TURBO_LIKELY
#if (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__clang__)
#if defined(__cplusplus)
#define TURBO_LIKELY(x)   __builtin_expect(!!(x), true)
#define TURBO_UNLIKELY(x) __builtin_expect(!!(x), false)
#else
#define TURBO_LIKELY(x)   __builtin_expect(!!(x), 1)
#define TURBO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#else
#define TURBO_LIKELY(x)   (x)
#define TURBO_UNLIKELY(x) (x)
#endif
#endif


// TURBO_ATTRIBUTE_INITIAL_EXEC
//
// Tells the compiler to use "initial-exec" mode for a thread-local variable.
// See http://people.redhat.com/drepper/tls.pdf for the gory details.
#if TURBO_HAVE_ATTRIBUTE(tls_model) || (defined(__GNUC__) && !defined(__clang__))
#define TURBO_ATTRIBUTE_INITIAL_EXEC __attribute__((tls_model("initial-exec")))
#else
#define TURBO_ATTRIBUTE_INITIAL_EXEC
#endif

#ifndef TURBO_THREAD_LOCAL
#ifdef TURBO_PLATFORM_WINDOWS
#define TURBO_THREAD_LOCAL __declspec(thread)
#else
#define TURBO_THREAD_LOCAL __thread
#endif
#endif

#endif // TURBO_PLATFORM_CONFIG_ATTRIBUTE_VARIABLE_H_
