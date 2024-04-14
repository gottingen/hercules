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

#ifndef TURBO_PLATFORM_CONFIG_ATTRIBUTE_FUNCTION_H_
#define TURBO_PLATFORM_CONFIG_ATTRIBUTE_FUNCTION_H_

#include "turbo/platform/config/compiler_traits.h"

// TURBO_ATTRIBUTE_NO_TAIL_CALL
//
// Prevents the compiler from optimizing away stack frames for functions which
// end in a call to another function.
#if TURBO_HAVE_ATTRIBUTE(disable_tail_calls)
#define TURBO_HAVE_ATTRIBUTE_NO_TAIL_CALL 1
#define TURBO_ATTRIBUTE_NO_TAIL_CALL __attribute__((disable_tail_calls))
#elif defined(__GNUC__) && !defined(__clang__) && !defined(__e2k__)
#define TURBO_HAVE_ATTRIBUTE_NO_TAIL_CALL 1
#define TURBO_ATTRIBUTE_NO_TAIL_CALL                                           \
  __attribute__((optimize("no-optimize-sibling-calls")))
#else
#define TURBO_ATTRIBUTE_NO_TAIL_CALL
#define TURBO_HAVE_ATTRIBUTE_NO_TAIL_CALL 0
#endif

/**
 * @ingroup turbo_function_macro
 * @brief This macro is used to mark a function as hot.
 * @details The `TURBO_HOT` macro is used to mark a function as hot. This
 *          attribute is used to inform the compiler that a function is a hot
 *          spot in the program. The function is optimized more aggressively
 *          and on many platforms special calling sequences are used. For
 *          example, on ARM targets a function marked as hot is expected to
 *          be called more often than a cold function. This attribute is
 *          often used to mark functions that are called frequently inside
 *          loops.
 * @code
 *          int foo() TURBO_HOT;
 * @endcode
 * @note    This attribute is not implemented in GCC versions earlier than
 *          4.3.0.
 * @note    This attribute is not implemented in Clang versions earlier than
 *          3.0.
 * @note    This attribute is not implemented in MSVC.
 * @note    This attribute is not implemented in ICC.
 * @note    This attribute is not implemented in IBM XL C/C++.
 * @note    This attribute is not implemented in TI C/C++.
 * @note    This attribute is not implemented in ARM C/C++.
 * @note    This attribute is not implemented in TI C/C++.
 * @note    This attribute is not implemented in TI C/C++.
 * @note    This attribute is not implemented in TI C/C++.
 */
#if TURBO_HAVE_ATTRIBUTE(hot) || (defined(__GNUC__) && !defined(__clang__))
#define TURBO_HOT __attribute__((hot))
#else
#define TURBO_HOT
#endif

#if TURBO_HAVE_ATTRIBUTE(cold) || (defined(__GNUC__) && !defined(__clang__))
#define TURBO_COLD __attribute__((cold))
#else
#define TURBO_COLD
#endif

// TURBO_ATTRIBUTE_TRIVIAL_ABI
// Indicates that a type is "trivially relocatable" -- meaning it can be
// relocated without invoking the constructor/destructor, using a form of move
// elision.
//
// From a memory safety point of view, putting aside destructor ordering, it's
// safe to apply TURBO_ATTRIBUTE_TRIVIAL_ABI if an object's location
// can change over the course of its lifetime: if a constructor can be run one
// place, and then the object magically teleports to another place where some
// methods are run, and then the object teleports to yet another place where it
// is destroyed. This is notably not true for self-referential types, where the
// move-constructor must keep the self-reference up to date. If the type changed
// location without invoking the move constructor, it would have a dangling
// self-reference.
//
// The use of this teleporting machinery means that the number of paired
// move/destroy operations can change, and so it is a bad idea to apply this to
// a type meant to count the number of moves.
//
// Warning: applying this can, rarely, break callers. Objects passed by value
// will be destroyed at the end of the call, instead of the end of the
// full-expression containing the call. In addition, it changes the ABI
// of functions accepting this type by value (e.g. to pass in registers).
//
// See also the upstream documentation:
// https://clang.llvm.org/docs/AttributeReference.html#trivial-abi
//

#if TURBO_HAVE_CPP_ATTRIBUTE(clang::trivial_abi)
#define TURBO_ATTRIBUTE_TRIVIAL_ABI [[clang::trivial_abi]]
#define TURBO_HAVE_ATTRIBUTE_TRIVIAL_ABI 1
#elif TURBO_HAVE_ATTRIBUTE(trivial_abi)
#define TURBO_ATTRIBUTE_TRIVIAL_ABI __attribute__((trivial_abi))
#define TURBO_HAVE_ATTRIBUTE_TRIVIAL_ABI 1
#else
#define TURBO_ATTRIBUTE_TRIVIAL_ABI
#endif

// TURBO_CONST_INIT
//
// A variable declaration annotated with the `TURBO_CONST_INIT` attribute will
// not compile (on supported platforms) unless the variable has a constant
// initializer. This is useful for variables with static and thread storage
// duration, because it guarantees that they will not suffer from the so-called
// "static init order fiasco".
//
// This attribute must be placed on the initializing declaration of the
// variable. Some compilers will give a -Wmissing-constinit warning when this
// attribute is placed on some other declaration but missing from the
// initializing declaration.
//
// In some cases (notably with thread_local variables), `TURBO_CONST_INIT` can
// also be used in a non-initializing declaration to tell the compiler that a
// variable is already initialized, reducing overhead that would otherwise be
// incurred by a hidden guard variable. Thus annotating all declarations with
// this attribute is recommended to potentially enhance optimization.
//
// Example:
//
//   class MyClass {
//    public:
//     TURBO_CONST_INIT static MyType my_var;
//   };
//
//   TURBO_CONST_INIT MyType MyClass::my_var = MakeMyType(...);
//
// For code or headers that are assured to only build with C++20 and up, prefer
// just using the standard `constinit` keyword directly over this macro.
//
// Note that this attribute is redundant if the variable is declared constexpr.
#if defined(__cpp_constinit) && __cpp_constinit >= 201907L
#define TURBO_CONST_INIT constinit
#elif TURBO_HAVE_CPP_ATTRIBUTE(clang::require_constant_initialization)
#define TURBO_CONST_INIT [[clang::require_constant_initialization]]
#else
#define TURBO_CONST_INIT
#endif

// TURBO_FUNC_ALIGN
//
// Tells the compiler to align the function start at least to certain
// alignment boundary
#if TURBO_HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define TURBO_FUNC_ALIGN(bytes) __attribute__((aligned(bytes)))
#else
#define TURBO_FUNC_ALIGN(bytes)
#endif

// TURBO_MUST_USE_RESULT
//
// Tells the compiler to warn about unused results.
//
// For code or headers that are assured to only build with C++17 and up, prefer
// just using the standard `[[nodiscard]]` directly over this macro.
//
// When annotating a function, it must appear as the first part of the
// declaration or definition. The compiler will warn if the return value from
// such a function is unused:
//
//   TURBO_MUST_USE_RESULT Sprocket* AllocateSprocket();
//   AllocateSprocket();  // Triggers a warning.
//
// When annotating a class, it is equivalent to annotating every function which
// returns an instance.
//
//   class TURBO_MUST_USE_RESULT Sprocket {};
//   Sprocket();  // Triggers a warning.
//
//   Sprocket MakeSprocket();
//   MakeSprocket();  // Triggers a warning.
//
// Note that references and pointers are not instances:
//
//   Sprocket* SprocketPointer();
//   SprocketPointer();  // Does *not* trigger a warning.
//
// TURBO_MUST_USE_RESULT allows using cast-to-void to suppress the unused result
// warning. For that, warn_unused_result is used only for clang but not for gcc.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66425
//
// Note: past advice was to place the macro after the argument list.
//
// TODO(b/176172494): Use TURBO_HAVE_CPP_ATTRIBUTE(nodiscard) when all code is
// compliant with the stricter [[nodiscard]].

/**
 * @ingroup turbo_function_macro
 * @brief This macro is used to mark a function as must use result.
 * @details The `TURBO_MUST_USE_RESULT` macro is used to mark a function as must
 *          use result. This attribute is used to inform the compiler that a
 *          function must be used. The function is optimized more aggressively
 *          and on many platforms special calling sequences are used. For
 *          example, on ARM targets a function marked as must use result is
 *          expected to be called more often than a cold function. This
 *          attribute is often used to mark functions that are called
 *          frequently inside loops.
 * @code
 *          TURBO_MUST_USE_RESULT int foo();
 * @endcode
 * @note    after c++17, use [[nodiscard]] instead.
 */
#if defined(__clang__) && TURBO_HAVE_ATTRIBUTE(warn_unused_result)
#define TURBO_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define TURBO_MUST_USE_RESULT
#endif

#ifndef TURBO_MAYBE_UNUSED
#if TURBO_HAVE_ATTRIBUTE(unused)
#define TURBO_MAYBE_UNUSED __attribute__((unused))
#else
#define TURBO_MAYBE_UNUSED
#endif
#endif // TURBO_MAYBE_UNUSED

/**
 * @ingroup turbo_function_macro
 * @brief This macro is used to force inline a function.
 * @details The `TURBO_FORCE_INLINE` macro is used to force inline a function.
 *          This attribute is used to inform the compiler that a function is a
 *          hot spot in the program. The function is optimized more aggressively
 *          and on many platforms special calling sequences are used. For
 *          example, on ARM targets a function marked as hot is expected to
 *          be called more often than a cold function. This attribute is
 *          often used to mark functions that are called frequently inside
 *          loops.
 * @code
 *          TUROB_FORCE_INLINE int foo();
 * @endcode
 * @note    This attribute is not implemented in GCC versions earlier than
 *         3.1.0. This attribute is not implemented in Clang versions earlier
 *         than 3.0.0. This attribute is not implemented in MSVC. This
 *         attribute is not implemented in ICC. This attribute is not
 *         implemented in IBM XL C/C++. This attribute is not implemented in
 *         TI C/C++. This attribute is not implemented in ARM C/C++. This
 *         attribute is not implemented in TI C/C++. This attribute is not
 *         implemented in TI C/C++.
 */
#ifndef TURBO_FORCE_INLINE
#if defined(TURBO_COMPILER_MSVC)
#define TURBO_FORCE_INLINE_SUPPORTED 1
#define TURBO_FORCE_INLINE __forceinline
#elif defined(TURBO_COMPILER_GNUC) &&                                          \
        (((__GNUC__ * 100) + __GNUC_MINOR__) >= 301) ||                        \
    defined(TURBO_COMPILER_CLANG)
#define TURBO_FORCE_INLINE_SUPPORTED 1
#if defined(__cplusplus)
#define TURBO_FORCE_INLINE inline __attribute__((always_inline))
#else
#define TURBO_FORCE_INLINE __inline__ __attribute__((always_inline))
#endif
#else
#if defined(__cplusplus)
#define TURBO_FORCE_INLINE inline
#else
#define TURBO_FORCE_INLINE __inline
#endif
#endif
#endif

#if defined(TURBO_COMPILER_GNUC) &&                                            \
        (((__GNUC__ * 100) + __GNUC_MINOR__) >= 301) ||                        \
    defined(TURBO_COMPILER_CLANG)
#define TURBO_PREFIX_FORCE_INLINE inline
#define TURBO_POSTFIX_FORCE_INLINE __attribute__((always_inline))
#else
#define TURBO_PREFIX_FORCE_INLINE inline
#define TURBO_POSTFIX_FORCE_INLINE
#endif

/**
 * @ingroup turbo_function_macro
 * @brief This macro is used to force inline a lambda function. Force inlining
 *        a lambda can be useful to reduce overhead in situations where a lambda
 *        may may only be called once, or inlining allows the compiler to apply
 *        other optimizations that wouldn't otherwise be possible.
 *
 *        The ability to force inline a lambda is currently only available on a
 *        subset of compilers.
 *
 *        Example usage:
 *        @code {.cpp}
 *        auto lambdaFunction = []() TURBO_FORCE_INLINE_LAMBDA
 *        {
 *        };
 *        @endcode
 */
#ifndef TURBO_FORCE_INLINE_LAMBDA
#if defined(TURBO_COMPILER_GNUC) || defined(TURBO_COMPILER_CLANG)
#define TURBO_FORCE_INLINE_LAMBDA __attribute__((always_inline))
#else
#define TURBO_FORCE_INLINE_LAMBDA
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_NO_INLINE             // Used as a prefix.
// TURBO_PREFIX_NO_INLINE      // You should need this only for unusual
// compilers. TURBO_POSTFIX_NO_INLINE     // You should need this only for
// unusual compilers.
//
// Example usage:
//     TURBO_NO_INLINE        void Foo();                       //
//     Implementation elsewhere. TURBO_PREFIX_NO_INLINE void Foo()
//     TURBO_POSTFIX_NO_INLINE;  // Implementation elsewhere.
//
// That this declaration is incompatbile with C++ 'inline' and any
// variant of TURBO_FORCE_INLINE.
//
// To disable inline usage under VC++ priof to VS2005, you need to use this:
//    #pragma inline_depth(0) // Disable inlining.
//    void Foo() { ... }
//    #pragma inline_depth()  // Restore to default.
//
// Since there is no easy way to disable inlining on a function-by-function
// basis in VC++ prior to VS2005, the best strategy is to write
// platform-specific #ifdefs in the code or to disable inlining for a given
// module and enable functions individually with TURBO_FORCE_INLINE.
//
#ifndef TURBO_NO_INLINE
#if defined(TURBO_COMPILER_MSVC) &&                                            \
    (TURBO_COMPILER_VERSION >= 1400) // If VC8 (VS2005) or later...
#define TURBO_NO_INLINE __declspec(noinline)
#define TURBO_NO_INLINE_SUPPORTED 1
#elif defined(TURBO_COMPILER_MSVC)
#define TURBO_NO_INLINE
#else
#define TURBO_NO_INLINE_SUPPORTED 1
#define TURBO_NO_INLINE __attribute__((noinline))
#endif
#endif

#if defined(TURBO_COMPILER_MSVC) &&                                            \
    (TURBO_COMPILER_VERSION >= 1400) // If VC8 (VS2005) or later...
#define TURBO_PREFIX_NO_INLINE __declspec(noinline)
#define TURBO_POSTFIX_NO_INLINE
#elif defined(TURBO_COMPILER_MSVC)
#define TURBO_PREFIX_NO_INLINE
#define TURBO_POSTFIX_NO_INLINE
#else
#define TURBO_PREFIX_NO_INLINE
#define TURBO_POSTFIX_NO_INLINE __attribute__((noinline))
#endif

// ------------------------------------------------------------------------
// TURBO_CURRENT_FUNCTION
//
// Provides a consistent way to get the current function name as a macro
// like the __FILE__ and __LINE__ macros work. The C99 standard specifies
// that __func__ be provided by the compiler, but most compilers don't yet
// follow that convention. However, many compilers have an alternative.
//
// We also define TURBO_CURRENT_FUNCTION_SUPPORTED for when it is not possible
// to have TURBO_CURRENT_FUNCTION work as expected.
//
// Defined inside a function because otherwise the macro might not be
// defined and code below might not compile. This happens with some
// compilers.
//
#ifndef TURBO_CURRENT_FUNCTION
#if defined __GNUC__ || (defined __ICC && __ICC >= 600)
#define TURBO_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
#define TURBO_CURRENT_FUNCTION __FUNCSIG__
#elif (defined __INTEL_COMPILER && __INTEL_COMPILER >= 600) || (defined __IBMCPP__ && __IBMCPP__ >= 500) || (defined CS_UNDEFINED_STRING && CS_UNDEFINED_STRING >= 0x4200)
#define TURBO_CURRENT_FUNCTION __FUNCTION__
#elif defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901
#define TURBO_CURRENT_FUNCTION __func__
#else
#define TURBO_CURRENT_FUNCTION "(unknown function)"
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_WEAK
// TURBO_WEAK_SUPPORTED -- defined as 0 or 1.
//
// GCC
// The weak attribute causes the declaration to be emitted as a weak
// symbol rather than a global. This is primarily useful in defining
// library functions which can be overridden in user code, though it
// can also be used with non-function declarations.
//
// VC++
// At link time, if multiple definitions of a COMDAT are seen, the linker
// picks one and discards the rest. If the linker option /OPT:REF
// is selected, then COMDAT elimination will occur to remove all the
// unreferenced data items in the linker output.
//
// Example usage:
//    TURBO_WEAK void Function();
//
#ifndef TURBO_WEAK
#if defined(_MSC_VER) && (_MSC_VER >= 1300) // If VC7.0 and later
#define TURBO_WEAK __declspec(selectany)
#define TURBO_WEAK_SUPPORTED 1
#elif defined(_MSC_VER) || (defined(__GNUC__) && defined(__CYGWIN__))
#define TURBO_WEAK
#define TURBO_WEAK_SUPPORTED 0
#elif defined(TURBO_COMPILER_ARM)  // Arm brand compiler for ARM CPU
#define TURBO_WEAK __weak
#define TURBO_WEAK_SUPPORTED 1
#else                           // GCC and IBM compilers, others.
#define TURBO_WEAK __attribute__((weak))
#define TURBO_WEAK_SUPPORTED 1
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_PURE
//
// This acts the same as the GCC __attribute__ ((pure)) directive and is
// implemented simply as a wrapper around it to allow portable usage of
// it and to take advantage of it if and when it appears in other compilers.
//
// A "pure" function is one that has no effects except its return value and
// its return value is a function of only the function's parameters or
// non-volatile global variables. Any parameter or global variable access
// must be read-only. Loop optimization and subexpression elimination can be
// applied to such functions. A common example is strlen(): Given identical
// inputs, the function's return value (its only effect) is invariant across
// multiple invocations and thus can be pulled out of a loop and called but once.
//
// Example usage:
//    TURBO_PURE void Function();
//
#ifndef TURBO_PURE
#if defined(TURBO_COMPILER_GNUC)
#define TURBO_PURE __attribute__((pure))
#elif defined(TURBO_COMPILER_ARM)  // Arm brand compiler for ARM CPU
#define TURBO_PURE __pure
#else
#define TURBO_PURE
#endif
#endif


// TURBO_DLL
//
// When building Turbo as a DLL, this macro expands to `__declspec(dllexport)`
// so we can annotate symbols appropriately as being exported. When used in
// headers consuming a DLL, this macro expands to `__declspec(dllimport)` so
// that consumers know the symbol is defined inside the DLL. In all other cases,
// the macro expands to nothing.
#if defined(_MSC_VER)
#if defined(TURBO_BUILD_DLL)
#define TURBO_DLL __declspec(dllexport)
#elif defined(TURBO_CONSUME_DLL)
#define TURBO_DLL __declspec(dllimport)
#else
#define TURBO_DLL
#endif
#else
#define TURBO_DLL
//#define TURBO_DLL __attribute__ ((visibility("default")))
#endif  // defined(_MSC_VER)

#ifndef TURBO_HIDDEN
#if defined(_MSC_VER)
#define TURBO_HIDDEN
#elif defined(__CYGWIN__)
#define TURBO_HIDDEN
#elif (defined(__GNUC__) && (__GNUC__ >= 4)) || TURBO_HAVE_ATTRIBUTE(visibility)
#define TURBO_HIDDEN    __attribute__ ((visibility("hidden")))
#else
#define TURBO_HIDDEN
#endif
#endif  // TURBO_HIDDEN

#ifndef TURBO_INLINE_VISIBILITY
#define TURBO_INLINE_VISIBILITY  TURBO_HIDDEN TURBO_FORCE_INLINE
#endif  // TURBO_INLINE_VISIBILITY


// ------------------------------------------------------------------------
// TURBO_CARRIES_DEPENDENCY
//
// Wraps the C++11 carries_dependency attribute
// http://en.cppreference.com/w/cpp/language/attributes
// http://blog.aaronballman.com/2011/09/understanding-attributes/
//
// Example usage:
//     TURBO_CARRIES_DEPENDENCY int* SomeFunction()
//         { return &mX; }
//
//
#if !defined(TURBO_CARRIES_DEPENDENCY)
#if defined(TURBO_COMPILER_NO_CARRIES_DEPENDENCY)
#define TURBO_CARRIES_DEPENDENCY
#else
#define TURBO_CARRIES_DEPENDENCY [[carries_dependency]]
#endif
#endif

// TURBO_NONNULL
//
// Tells the compiler either (a) that a particular function parameter
// should be a non-null pointer, or (b) that all pointer arguments should
// be non-null.
//
// Note: As the GCC manual states, "[s]ince non-static C++ methods
// have an implicit 'this' argument, the arguments of such methods
// should be counted from two, not one."
//
// Args are indexed starting at 1.
//
// For non-static class member functions, the implicit `this` argument
// is arg 1, and the first explicit argument is arg 2. For static class member
// functions, there is no implicit `this`, and the first explicit argument is
// arg 1.
//
// Example:
//
//   /* arg_a cannot be null, but arg_b can */
//   void Function(void* arg_a, void* arg_b) TURBO_NONNULL(1);
//
//   class C {
//     /* arg_a cannot be null, but arg_b can */
//     void Method(void* arg_a, void* arg_b) TURBO_NONNULL(2);
//
//     /* arg_a cannot be null, but arg_b can */
//     static void StaticMethod(void* arg_a, void* arg_b)
//     TURBO_NONNULL(1);
//   };
//
// If no arguments are provided, then all pointer arguments should be non-null.
//
//  /* No pointer arguments may be null. */
//  void Function(void* arg_a, void* arg_b, int arg_c) TURBO_NONNULL();
//
// NOTE: The GCC nonnull attribute actually accepts a list of arguments, but
// TURBO_NONNULL does not.
#if TURBO_HAVE_ATTRIBUTE(nonnull) || (defined(__GNUC__) && !defined(__clang__))
#define TURBO_NONNULL(arg_index) __attribute__((nonnull(arg_index)))
#else
#define TURBO_NONNULL(...)
#endif

// ------------------------------------------------------------------------
// TURBO_NORETURN
//
// Wraps the C++11 noreturn attribute. See TURBO_COMPILER_NO_NORETURN
// http://en.cppreference.com/w/cpp/language/attributes
// http://msdn.microsoft.com/en-us/library/k6ktzx3s%28v=vs.80%29.aspx
// http://blog.aaronballman.com/2011/09/understanding-attributes/
//
// Example usage:
//     TURBO_NORETURN void SomeFunction()
//         { throw "error"; }
//
#if !defined(TURBO_NORETURN)
#if defined(TURBO_COMPILER_MSVC) && (TURBO_COMPILER_VERSION >= 1300) // VS2003 (VC7) and later
#define TURBO_NORETURN __declspec(noreturn)
#elif defined(TURBO_COMPILER_NO_NORETURN)
#define TURBO_NORETURN
#else
#define TURBO_NORETURN __attribute__((noreturn))
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_EMPTY
//
// Allows for a null statement, usually for the purpose of avoiding compiler warnings.
//
// Example usage:
//    #ifdef TURBO_DEBUG
//        #define MyDebugPrintf(x, y) printf(x, y)
//    #else
//        #define MyDebugPrintf(x, y)  TURBO_EMPTY
//    #endif
//
#ifndef TURBO_EMPTY
#define TURBO_EMPTY (void)0
#endif

// TURBO_PRINTF_ATTRIBUTE
// TURBO_SCANF_ATTRIBUTE
//
// Tells the compiler to perform `printf` format string checking if the
// compiler supports it; see the 'format' attribute in
// <https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/Function-Attributes.html>.
//
// Note: As the GCC manual states, "[s]ince non-static C++ methods
// have an implicit 'this' argument, the arguments of such methods
// should be counted from two, not one."
#if TURBO_HAVE_ATTRIBUTE(format) || (defined(__GNUC__) && !defined(__clang__))
#define TURBO_PRINTF_ATTRIBUTE(string_index, first_to_check) \
      __attribute__((__format__(__printf__, string_index, first_to_check)))
#define TURBO_SCANF_ATTRIBUTE(string_index, first_to_check) \
      __attribute__((__format__(__scanf__, string_index, first_to_check)))
#else
#define TURBO_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define TURBO_SCANF_ATTRIBUTE(string_index, first_to_check)
#endif

// TURBO_HAVE_ATTRIBUTE_SECTION
//
// Indicates whether labeled sections are supported. Weak symbol support is
// a prerequisite. Labeled sections are not supported on Darwin/iOS.
#ifdef TURBO_HAVE_ATTRIBUTE_SECTION
#error TURBO_HAVE_ATTRIBUTE_SECTION cannot be directly set
#elif (TURBO_HAVE_ATTRIBUTE(section) ||                \
       (defined(__GNUC__) && !defined(__clang__))) && \
    !defined(__APPLE__) && TURBO_WEAK_SUPPORTED
#define TURBO_HAVE_ATTRIBUTE_SECTION 1

// TURBO_ATTRIBUTE_SECTION
//
// Tells the compiler/linker to put a given function into a section and define
// `__start_ ## name` and `__stop_ ## name` symbols to bracket the section.
// This functionality is supported by GNU linker.  Any function annotated with
// `TURBO_ATTRIBUTE_SECTION` must not be inlined, or it will be placed into
// whatever section its caller is placed into.
//
#ifndef TURBO_ATTRIBUTE_SECTION
#define TURBO_ATTRIBUTE_SECTION(name) \
  __attribute__((section(#name))) __attribute__((noinline))
#endif

// TURBO_ATTRIBUTE_SECTION_VARIABLE
//
// Tells the compiler/linker to put a given variable into a section and define
// `__start_ ## name` and `__stop_ ## name` symbols to bracket the section.
// This functionality is supported by GNU linker.
#ifndef TURBO_ATTRIBUTE_SECTION_VARIABLE
#ifdef _AIX
// __attribute__((section(#name))) on AIX is achived by using the `.csect` psudo
// op which includes an additional integer as part of its syntax indcating
// alignment. If data fall under different alignments then you might get a
// compilation error indicating a `Section type conflict`.
#define TURBO_ATTRIBUTE_SECTION_VARIABLE(name)
#else
#define TURBO_ATTRIBUTE_SECTION_VARIABLE(name) __attribute__((section(#name)))
#endif
#endif

// TURBO_DECLARE_ATTRIBUTE_SECTION_VARS
//
// A weak section declaration to be used as a global declaration
// for TURBO_ATTRIBUTE_SECTION_START|STOP(name) to compile and link
// even without functions with TURBO_ATTRIBUTE_SECTION(name).
// TURBO_DEFINE_ATTRIBUTE_SECTION should be in the exactly one file; it's
// a no-op on ELF but not on Mach-O.
//
#ifndef TURBO_DECLARE_ATTRIBUTE_SECTION_VARS
#define TURBO_DECLARE_ATTRIBUTE_SECTION_VARS(name)   \
  extern char __start_##name[] TURBO_WEAK; \
  extern char __stop_##name[] TURBO_WEAK
#endif
#ifndef TURBO_DEFINE_ATTRIBUTE_SECTION_VARS
#define TURBO_INIT_ATTRIBUTE_SECTION_VARS(name)
#define TURBO_DEFINE_ATTRIBUTE_SECTION_VARS(name)
#endif

// TURBO_ATTRIBUTE_SECTION_START
//
// Returns `void*` pointers to start/end of a section of code with
// functions having TURBO_ATTRIBUTE_SECTION(name).
// Returns 0 if no such functions exist.
// One must TURBO_DECLARE_ATTRIBUTE_SECTION_VARS(name) for this to compile and
// link.
//
#define TURBO_ATTRIBUTE_SECTION_START(name) \
  (reinterpret_cast<void *>(__start_##name))
#define TURBO_ATTRIBUTE_SECTION_STOP(name) \
  (reinterpret_cast<void *>(__stop_##name))

#else  // !TURBO_HAVE_ATTRIBUTE_SECTION

#define TURBO_HAVE_ATTRIBUTE_SECTION 0

// provide dummy definitions
#define TURBO_ATTRIBUTE_SECTION(name)
#define TURBO_ATTRIBUTE_SECTION_VARIABLE(name)
#define TURBO_INIT_ATTRIBUTE_SECTION_VARS(name)
#define TURBO_DEFINE_ATTRIBUTE_SECTION_VARS(name)
#define TURBO_DECLARE_ATTRIBUTE_SECTION_VARS(name)
#define TURBO_ATTRIBUTE_SECTION_START(name) (reinterpret_cast<void *>(0))
#define TURBO_ATTRIBUTE_SECTION_STOP(name) (reinterpret_cast<void *>(0))

#endif  // TURBO_ATTRIBUTE_SECTION


// TURBO_XRAY_ALWAYS_INSTRUMENT, TURBO_XRAY_NEVER_INSTRUMENT, TURBO_XRAY_LOG_ARGS
//
// We define the TURBO_XRAY_ALWAYS_INSTRUMENT and TURBO_XRAY_NEVER_INSTRUMENT
// macro used as an attribute to mark functions that must always or never be
// instrumented by XRay. Currently, this is only supported in Clang/LLVM.
//
// For reference on the LLVM XRay instrumentation, see
// http://llvm.org/docs/XRay.html.
//
// A function with the XRAY_ALWAYS_INSTRUMENT macro attribute in its declaration
// will always get the XRay instrumentation sleds. These sleds may introduce
// some binary size and runtime overhead and must be used sparingly.
//
// These attributes only take effect when the following conditions are met:
//
//   * The file/target is built in at least C++11 mode, with a Clang compiler
//     that supports XRay attributes.
//   * The file/target is built with the -fxray-instrument flag set for the
//     Clang/LLVM compiler.
//   * The function is defined in the translation unit (the compiler honors the
//     attribute in either the definition or the declaration, and must match).
//
// There are cases when, even when building with XRay instrumentation, users
// might want to control specifically which functions are instrumented for a
// particular build using special-case lists provided to the compiler. These
// special case lists are provided to Clang via the
// -fxray-always-instrument=... and -fxray-never-instrument=... flags. The
// attributes in source take precedence over these special-case lists.
//
// To disable the XRay attributes at build-time, users may define
// TURBO_NO_XRAY_ATTRIBUTES. Do NOT define TURBO_NO_XRAY_ATTRIBUTES on specific
// packages/targets, as this may lead to conflicting definitions of functions at
// link-time.
//
// XRay isn't currently supported on Android:
// https://github.com/android/ndk/issues/368
#if TURBO_HAVE_CPP_ATTRIBUTE(clang::xray_always_instrument) && \
    !defined(TURBO_NO_XRAY_ATTRIBUTES) && !defined(__ANDROID__)
#define TURBO_XRAY_ALWAYS_INSTRUMENT [[clang::xray_always_instrument]]
#define TURBO_XRAY_NEVER_INSTRUMENT [[clang::xray_never_instrument]]
#if TURBO_HAVE_CPP_ATTRIBUTE(clang::xray_log_args)
#define TURBO_XRAY_LOG_ARGS(N) \
  [[clang::xray_always_instrument, clang::xray_log_args(N)]]
#else
#define TURBO_XRAY_LOG_ARGS(N) [[clang::xray_always_instrument]]
#endif
#else
#define TURBO_XRAY_ALWAYS_INSTRUMENT
#define TURBO_XRAY_NEVER_INSTRUMENT
#define TURBO_XRAY_LOG_ARGS(N)
#endif

// TURBO_ATTRIBUTE_REINITIALIZES
//
// Indicates that a member function reinitializes the entire object to a known
// state, independent of the previous state of the object.
//
// The clang-tidy check bugprone-use-after-move allows member functions marked
// with this attribute to be called on objects that have been moved from;
// without the attribute, this would result in a use-after-move warning.
#if TURBO_HAVE_CPP_ATTRIBUTE(clang::reinitializes)
#define TURBO_ATTRIBUTE_REINITIALIZES [[clang::reinitializes]]
#else
#define TURBO_ATTRIBUTE_REINITIALIZES
#endif


// TURBO_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC
//
// Support for aligning the stack on 32-bit x86.
#if TURBO_HAVE_ATTRIBUTE(force_align_arg_pointer) || \
    (defined(__GNUC__) && !defined(__clang__))
#if defined(__i386__)
#define TURBO_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC \
  __attribute__((force_align_arg_pointer))
#define TURBO_REQUIRE_STACK_ALIGN_TRAMPOLINE (0)
#elif defined(__x86_64__)
#define TURBO_REQUIRE_STACK_ALIGN_TRAMPOLINE (1)
#define TURBO_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC
#else  // !__i386__ && !__x86_64
#define TURBO_REQUIRE_STACK_ALIGN_TRAMPOLINE (0)
#define TURBO_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC
#endif  // __i386__
#else
#define TURBO_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC
#define TURBO_REQUIRE_STACK_ALIGN_TRAMPOLINE (0)
#endif


// TURBO_BAD_CALL_IF()
//
// Used on a function overload to trap bad calls: any call that matches the
// overload will cause a compile-time error. This macro uses a clang-specific
// "enable_if" attribute, as described at
// https://clang.llvm.org/docs/AttributeReference.html#enable-if
//
// Overloads which use this macro should be bracketed by
// `#ifdef TURBO_BAD_CALL_IF`.
//
// Example:
//
//   int isdigit(int c);
//   #ifdef TURBO_BAD_CALL_IF
//   int isdigit(int c)
//     TURBO_BAD_CALL_IF(c <= -1 || c > 255,
//                       "'c' must have the value of an unsigned char or EOF");
//   #endif // TURBO_BAD_CALL_IF
#if TURBO_HAVE_ATTRIBUTE(enable_if)
#define TURBO_BAD_CALL_IF(expr, msg) \
  __attribute__((enable_if(expr, "Bad call trap"), unavailable(msg)))
#endif


// TURBO_RETURNS_NONNULL
//
// Tells the compiler that a particular function never returns a null pointer.
#if TURBO_HAVE_ATTRIBUTE(returns_nonnull)
#define TURBO_RETURNS_NONNULL __attribute__((returns_nonnull))
#else
#define TURBO_RETURNS_NONNULL
#endif

#endif // TURBO_PLATFORM_CONFIG_ATTRIBUTE_FUNCTION_H_
