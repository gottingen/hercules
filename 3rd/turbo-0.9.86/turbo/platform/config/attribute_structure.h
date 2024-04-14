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

#ifndef TURBO_PLATFORM_CONFIG_ATTRIBUTE_STRUCTURE_H_
#define TURBO_PLATFORM_CONFIG_ATTRIBUTE_STRUCTURE_H_

#include "turbo/platform/config/compiler_traits.h"

// ------------------------------------------------------------------------
// TURBO_NON_COPYABLE
//
// This macro defines as a class as not being copy-constructable
// or assignable. This is useful for preventing class instances
// from being passed to functions by value, is useful for preventing
// compiler warnings by some compilers about the inability to
// auto-generate a copy constructor and assignment, and is useful
// for simply declaring in the interface that copy semantics are
// not supported by the class. Your class needs to have at least a
// default constructor when using this macro.
//
// Beware that this class works by declaring a private: section of
// the class in the case of compilers that don't support C++11 deleted
// functions.
//
// Note: With some pre-C++11 compilers (e.g. Green Hills), you may need
//       to manually define an instances of the hidden functions, even
//       though they are not used.
//
// Example usage:
//    class Widget {
//       Widget();
//       . . .
//       TURBO_NON_COPYABLE(Widget)
//    };
//
#if !defined(TURBO_NON_COPYABLE)
#define TURBO_NON_COPYABLE(TBClass_)               \
        TURBO_DISABLE_VC_WARNING(4822);	/* local class member function does not have a body	*/		\
        TBClass_(const TBClass_&) = delete;         \
        void operator=(const TBClass_&) = delete;	\
        TURBO_RESTORE_VC_WARNING();
#endif

#if !defined(TURBO_NON_MOVEABLE)
#define TURBO_NON_MOVEABLE(TBClass_)               \
        TURBO_DISABLE_VC_WARNING(4822);	/* local class member function does not have a body	*/		\
        TBClass_(TBClass_*&) = delete;         \
        void operator=(TBClass_&&) = delete;	\
        TURBO_RESTORE_VC_WARNING();
#endif

// ------------------------------------------------------------------------
// TURBO_OFFSETOF
// Implements a portable version of the non-standard offsetof macro.
//
// The offsetof macro is guaranteed to only work with POD types. However, we wish to use
// it for non-POD types but where we know that offsetof will still work for the cases
// in which we use it. GCC unilaterally gives a warning when using offsetof with a non-POD,
// even if the given usage happens to work. So we make a workaround version of offsetof
// here for GCC which has the same effect but tricks the compiler into not issuing the warning.
// The 65536 does the compiler fooling; the reinterpret_cast prevents the possibility of
// an overloaded operator& for the class getting in the way.
//
// Example usage:
//     struct A{ int x; int y; };
//     size_t n = TURBO_OFFSETOF(A, y);
//
#if defined(__GNUC__)                       // We can't use GCC 4's __builtin_offsetof because it mistakenly complains about non-PODs that are really PODs.
#define TURBO_OFFSETOF(struct_, member_)  ((size_t)(((uintptr_t)&reinterpret_cast<const volatile char&>((((struct_*)65536)->member_))) - 65536))
#else
#define TURBO_OFFSETOF(struct_, member_)  offsetof(struct_, member_)
#endif

// ------------------------------------------------------------------------
// TURBO_SIZEOF_MEMBER
// Implements a portable way to determine the size of a member.
//
// The TURBO_SIZEOF_MEMBER simply returns the size of a member within a class or struct; member
// access rules still apply. We offer two approaches depending on the compiler's support for non-static member
// initializers although most C++11 compilers support this.
//
// Example usage:
//     struct A{ int x; int y; };
//     size_t n = TURBO_SIZEOF_MEMBER(A, y);
//
#ifndef TURBO_COMPILER_NO_EXTENDED_SIZEOF
#define TURBO_SIZEOF_MEMBER(struct_, member_) (sizeof(struct_::member_))
#else
#define TURBO_SIZEOF_MEMBER(struct_, member_) (sizeof(((struct_*)0)->member_))
#endif

// We are going to use runtime dispatch.
#ifdef TURBO_PROCESSOR_X86_64
#ifdef __clang__
// clang does not have GCC push pop
// warning: clang attribute push can't be used within a namespace in clang up
// til 8.0 so TURBO_TARGET_REGION and TURBO_UNTARGET_REGION must be *outside* of a
// namespace.
#define TURBO_TARGET_REGION(T)                                                       \
  _Pragma(TURBO_STRINGIFY(                                                           \
      clang attribute push(__attribute__((target(T))), apply_to = function)))
#define TURBO_UNTARGET_REGION _Pragma("clang attribute pop")
#elif defined(__GNUC__)
// GCC is easier
#define TURBO_TARGET_REGION(T)                                                       \
  _Pragma("GCC push_options") _Pragma(TURBO_STRINGIFY(GCC target(T)))
#define TURBO_UNTARGET_REGION _Pragma("GCC pop_options")
#endif // clang then gcc

#endif // x86

// Default target region macros don't do anything.
#ifndef TURBO_TARGET_REGION
#define TURBO_TARGET_REGION(T)
#define TURBO_UNTARGET_REGION
#endif


#endif  // TURBO_PLATFORM_CONFIG_ATTRIBUTE_STRUCTURE_H_
