/*-----------------------------------------------------------------------------
 * config/compiler_traits.h
 *
 * Copyright (c) Electronic Arts Inc. All rights reserved.
 * Copyright (c) Jeff.Li
 *-----------------------------------------------------------------------------
 * Currently supported defines include:
 *    TURBO_CONCAT
 *    
 *    TURBO_COMPILER_IS_ANSIC
 *    TURBO_COMPILER_IS_C99
 *    TURBO_COMPILER_IS_C11
 *    TURBO_COMPILER_HAS_C99_TYPES
 *    TURBO_COMPILER_IS_CPLUSPLUS
 *    TURBO_COMPILER_MANAGED_CPP
 *    TURBO_COMPILER_INTMAX_SIZE
 *    TURBO_OFFSETOF
 *    TURBO_SIZEOF_MEMBER
 *
 *    TURBO_ALIGN_OF()
 *    TURBO_ALIGN_MAX_STATIC / TURBO_ALIGN_MAX_AUTOMATIC
 *    TURBO_ALIGN() / TURBO_PREFIX_ALIGN() / TURBO_POSTFIX_ALIGN()
 *    TURBO_ALIGNED()
 *    TURBO_PACKED()
 *
 *    TURBO_LIKELY()
 *    TURBO_UNLIKELY()
 *    TURBO_INIT_PRIORITY()
 *    TURBO_MAY_ALIAS()
 *    TURBO_ASSUME()
 *    TURBO_ANALYSIS_ASSUME()
 *    TURBO_PURE
 *    TURBO_WEAK
 *    TURBO_UNUSED()
 *    TURBO_EMPTY()
 *
 *    TURBO_WCHAR_T_NON_NATIVE
 *    TURBO_WCHAR_SIZE = <n bytes>
 *
 *    TURBO_RESTRICT
 *    TURBO_FORCE_INLINE / TURBO_PREFIX_FORCE_INLINE / TURBO_POSTFIX_FORCE_INLINE
 *    TURBO_NO_INLINE    / TURBO_PREFIX_NO_INLINE    / TURBO_POSTFIX_NO_INLINE
 *    TURBO_NO_VTABLE    / TURBO_CLASS_NO_VTABLE     / TURBO_STRUCT_NO_VTABLE
 *    TURBO_PASCAL
 *    TURBO_PASCAL_FUNC()
 *    TURBO_IMPORT
 *    TURBO_EXPORT
 *    TURBO_PRAGMA_ONCE_SUPPORTED
 *    TURBO_CONSTEXPR / TURBO_CONSTEXPR_OR_CONST
 *    TURBO_CONSTEXPR_IF
 *    TURBO_EXTERN_TEMPLATE
 *    TURBO_NOEXCEPT
 *    TURBO_NORETURN
 *    TURBO_CARRIES_DEPENDENCY
 *    TURBO_NON_COPYABLE / struct EANonCopyable
 *    TURBO_OPTIMIZE_OFF / TURBO_OPTIMIZE_ON
 *    TURBO_SIGNED_RIGHT_SHIFT_IS_UNSIGNED
 *
 *    TURBO_DISABLE_VC_WARNING    / TURBO_RESTORE_VC_WARNING / TURBO_DISABLE_ALL_VC_WARNINGS / TURBO_RESTORE_ALL_VC_WARNINGS
 *    TURBO_DISABLE_GCC_WARNING   / TURBO_RESTORE_GCC_WARNING
 *    TURBO_DISABLE_CLANG_WARNING / TURBO_RESTORE_CLANG_WARNING
 *    TURBO_DISABLE_SN_WARNING    / TURBO_RESTORE_SN_WARNING / TURBO_DISABLE_ALL_SN_WARNINGS / TURBO_RESTORE_ALL_SN_WARNINGS
 *    TURBO_DISABLE_GHS_WARNING   / TURBO_RESTORE_GHS_WARNING
 *    TURBO_DISABLE_EDG_WARNING   / TURBO_RESTORE_EDG_WARNING
 *    TURBO_DISABLE_CW_WARNING    / TURBO_RESTORE_CW_WARNING
 *
 *    TURBO_DISABLE_DEFAULT_CTOR
 *    TURBO_DISABLE_COPY_CTOR
 *    TURBO_DISABLE_MOVE_CTOR
 *    TURBO_DISABLE_ASSIGNMENT_OPERATOR
 *    TURBO_DISABLE_MOVE_OPERATOR
 *
 *  Todo:
 *    Find a way to reliably detect wchar_t size at preprocessor time and 
 *    implement it below for TURBO_WCHAR_SIZE.
 *
 *  Todo:
 *    Find out how to support TURBO_PASCAL and TURBO_PASCAL_FUNC for systems in
 *    which it hasn't yet been found out for.
 *---------------------------------------------------------------------------*/


#ifndef TURBO_PLATFORM_CONFIG_COMPILER_TRAITS_H_
#define TURBO_PLATFORM_CONFIG_COMPILER_TRAITS_H_

#include "turbo/platform/config/platform.h"
#include "turbo/platform/config/compiler.h"


    // TURBO_HAVE_FEATURE
    #ifndef TURBO_HAVE_FEATURE
        #if defined(__has_feature)
            #define TURBO_HAVE_FEATURE(x) __has_feature(x)
        #else
            #define TURBO_HAVE_FEATURE(x) 0
        #endif
    #endif

    // TURBO_HAVE_ATTRIBUTE
    //
    // A function-like feature checking macro that is a wrapper around
    // `__has_attribute`, which is defined by GCC 5+ and Clang and evaluates to a
    // nonzero constant integer if the attribute is supported or 0 if not.
    //
    // It evaluates to zero if `__has_attribute` is not defined by the compiler.
    //
    // GCC: https://gcc.gnu.org/gcc-5/changes.html
    // Clang: https://clang.llvm.org/docs/LanguageExtensions.html
    #ifndef TURBO_HAVE_ATTRIBUTE
        #ifdef __has_attribute
            #define TURBO_HAVE_ATTRIBUTE(x) __has_attribute(x)
        #else
            #define TURBO_HAVE_ATTRIBUTE(x) 0
        #endif
    #endif


    // TURBO_HAVE_BUILTIN()
    //
    // Checks whether the compiler supports a Clang Feature Checking Macro, and if
    // so, checks whether it supports the provided builtin function "x" where x
    // is one of the functions noted in
    // https://clang.llvm.org/docs/LanguageExtensions.html
    //
    // Note: Use this macro to avoid an extra level of #ifdef __has_builtin check.
    // http://releases.llvm.org/3.3/tools/clang/docs/LanguageExtensions.html
    #ifndef TURBO_HAVE_BUILTIN
        #ifdef __has_builtin
            #define TURBO_HAVE_BUILTIN(x) __has_builtin(x)
        #else
            #define TURBO_HAVE_BUILTIN(x) 0
        #endif
    #endif
    // TURBO_HAVE_CPP_ATTRIBUTE
    //
    // A function-like feature checking macro that accepts C++11 style attributes.
    // It's a wrapper around `__has_cpp_attribute`, defined by ISO C++ SD-6
    // (https://en.cppreference.com/w/cpp/experimental/feature_test). If we don't
    // find `__has_cpp_attribute`, will evaluate to 0.
    #if defined(__cplusplus) && defined(__has_cpp_attribute)
    // NOTE: requiring __cplusplus above should not be necessary, but
    // works around https://bugs.llvm.org/show_bug.cgi?id=23435.
    #define TURBO_HAVE_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
    #else
    #define TURBO_HAVE_CPP_ATTRIBUTE(x) 0
    #endif

	// Determine if this compiler is ANSI C compliant and if it is C99 compliant.
	#if defined(__STDC__)
		#define TURBO_COMPILER_IS_ANSIC 1    // The compiler claims to be ANSI C

		// Is the compiler a C99 compiler or equivalent?
		// From ISO/IEC 9899:1999:
		//    6.10.8 Predefined macro names
		//    __STDC_VERSION__ The integer constant 199901L. (150)
		//
		//    150) This macro was not specified in ISO/IEC 9899:1990 and was
		//    specified as 199409L in ISO/IEC 9899/AMD1:1995. The intention
		//    is that this will remain an integer constant of type long int
		//    that is increased with each revision of this International Standard.
		//
		#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
			#define TURBO_COMPILER_IS_C99 1
		#endif

 		// Is the compiler a C11 compiler?
 		// From ISO/IEC 9899:2011:
		//   Page 176, 6.10.8.1 (Predefined macro names) :
 		//   __STDC_VERSION__ The integer constant 201112L. (178)
		//
		#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
			#define TURBO_COMPILER_IS_C11 1
		#endif
	#endif

	// Some compilers (e.g. GCC) define __USE_ISOC99 if they are not
	// strictly C99 compilers (or are simply C++ compilers) but are set
	// to use C99 functionality. Metrowerks defines _MSL_C99 as 1 in
	// this case, but 0 otherwise.
	#if (defined(__USE_ISOC99) || (defined(_MSL_C99) && (_MSL_C99 == 1))) && !defined(TURBO_COMPILER_IS_C99)
		#define TURBO_COMPILER_IS_C99 1
	#endif

	// Metrowerks defines C99 types (e.g. intptr_t) instrinsically when in C99 mode (-lang C99 on the command line).
	#if (defined(_MSL_C99) && (_MSL_C99 == 1))
		#define TURBO_COMPILER_HAS_C99_TYPES 1
	#endif

	#if defined(__GNUC__)
		#if (((__GNUC__ * 100) + __GNUC_MINOR__) >= 302) // Also, GCC defines _HAS_C9X.
			#define TURBO_COMPILER_HAS_C99_TYPES 1 // The compiler is not necessarily a C99 compiler, but it defines C99 types.

			#ifndef __STDC_LIMIT_MACROS
				#define __STDC_LIMIT_MACROS 1
			#endif

			#ifndef __STDC_CONSTANT_MACROS
				#define __STDC_CONSTANT_MACROS 1    // This tells the GCC compiler that we want it to use its native C99 types.
			#endif
		#endif
	#endif

	#if defined(_MSC_VER) && (_MSC_VER >= 1600)
		#define TURBO_COMPILER_HAS_C99_TYPES 1
	#endif

	// ------------------------------------------------------------------------
	// TURBO_COMPILER_INTMAX_SIZE
	//
	// This is related to the concept of intmax_t uintmax_t, but is available
	// in preprocessor form as opposed to compile-time form. At compile-time
	// you can use intmax_t and uintmax_t to use the actual types.
	//
	#if defined(__GNUC__) && defined(__x86_64__)
		#define TURBO_COMPILER_INTMAX_SIZE 16  // intmax_t is __int128_t (GCC extension) and is 16 bytes.
	#else
		#define TURBO_COMPILER_INTMAX_SIZE 8   // intmax_t is int64_t and is 8 bytes.
	#endif

	// ------------------------------------------------------------------------
	// alignment expressions
	//
	// Here we define
	//    TURBO_ALIGN_OF(type)         // Returns size_t.
	//    TURBO_ALIGN_MAX_STATIC       // The max align value that the compiler will respect for TURBO_ALIGN for static data (global and static variables). Some compilers allow high values, some allow no more than 8. TURBO_ALIGN_MIN is assumed to be 1.
	//    TURBO_ALIGN_MAX_AUTOMATIC    // The max align value for automatic variables (variables declared as local to a function).
	//    TURBO_ALIGN(n)               // Used as a prefix. n is byte alignment, with being a power of two. Most of the time you can use this and avoid using TURBO_PREFIX_ALIGN/TURBO_POSTFIX_ALIGN.
	//    TURBO_ALIGNED(t, v, n)       // Type, variable, alignment. Used to align an instance. You should need this only for unusual compilers.
	//    TURBO_PACKED                 // Specifies that the given structure be packed (and not have its members aligned).
	//
	// Also we define the following for rare cases that it's needed.
	//    TURBO_PREFIX_ALIGN(n)        // n is byte alignment, with being a power of two. You should need this only for unusual compilers.
	//    TURBO_POSTFIX_ALIGN(n)       // Valid values for n are 1, 2, 4, 8, etc. You should need this only for unusual compilers.
	//
	// Example usage:
	//    size_t x = TURBO_ALIGN_OF(int);                                  Non-aligned equivalents.        Meaning
	//    TURBO_PREFIX_ALIGN(8) int x = 5;                                 int x = 5;                      Align x on 8 for compilers that require prefix attributes. Can just use TURBO_ALIGN instead.
	//    TURBO_ALIGN(8) int x;                                            int x;                          Align x on 8 for compilers that allow prefix attributes.
	//    int x TURBO_POSTFIX_ALIGN(8);                                    int x;                          Align x on 8 for compilers that require postfix attributes.
	//    int x TURBO_POSTFIX_ALIGN(8) = 5;                                int x = 5;                      Align x on 8 for compilers that require postfix attributes.
	//    int x TURBO_POSTFIX_ALIGN(8)(5);                                 int x(5);                       Align x on 8 for compilers that require postfix attributes.
	//    struct TURBO_PREFIX_ALIGN(8) X { int x; } TURBO_POSTFIX_ALIGN(8);   struct X { int x; };            Define X as a struct which is aligned on 8 when used.
	//    TURBO_ALIGNED(int, x, 8) = 5;                                    int x = 5;                      Align x on 8.
	//    TURBO_ALIGNED(int, x, 16)(5);                                    int x(5);                       Align x on 16.
	//    TURBO_ALIGNED(int, x[3], 16);                                    int x[3];                       Align x array on 16.
	//    TURBO_ALIGNED(int, x[3], 16) = { 1, 2, 3 };                      int x[3] = { 1, 2, 3 };         Align x array on 16.
	//    int x[3] TURBO_PACKED;                                           int x[3];                       Pack the 3 ints of the x array. GCC doesn't seem to support packing of int arrays.
	//    struct TURBO_ALIGN(32) X { int x; int y; };                      struct X { int x; };            Define A as a struct which is aligned on 32 when used.
	//    TURBO_ALIGN(32) struct X { int x; int y; } Z;                    struct X { int x; } Z;          Define A as a struct, and align the instance Z on 32.
	//    struct X { int x TURBO_PACKED; int y TURBO_PACKED; };               struct X { int x; int y; };     Pack the x and y members of struct X.
	//    struct X { int x; int y; } TURBO_PACKED;                         struct X { int x; int y; };     Pack the members of struct X.
	//    typedef TURBO_ALIGNED(int, int16, 16); int16 n16;                typedef int int16; int16 n16;   Define int16 as an int which is aligned on 16.
	//    typedef TURBO_ALIGNED(X, X16, 16); X16 x16;                      typedef X X16; X16 x16;         Define X16 as an X which is aligned on 16.

	#if !defined(TURBO_ALIGN_MAX)                              // If the user hasn't globally set an alternative value...
		#if defined(TURBO_PROCESSOR_ARM)                       // ARM compilers in general tend to limit automatic variables to 8 or less.
			#define TURBO_ALIGN_MAX_STATIC    1048576
			#define TURBO_ALIGN_MAX_AUTOMATIC       1          // Typically they support only built-in natural aligment types (both arm-eabi and apple-abi).
		#elif defined(TURBO_PLATFORM_APPLE)
			#define TURBO_ALIGN_MAX_STATIC    1048576
			#define TURBO_ALIGN_MAX_AUTOMATIC      16
		#else
			#define TURBO_ALIGN_MAX_STATIC    1048576          // Arbitrarily high value. What is the actual max?
			#define TURBO_ALIGN_MAX_AUTOMATIC 1048576
		#endif
	#endif

	// EDG intends to be compatible with GCC but has a bug whereby it
	// fails to support calling a constructor in an aligned declaration when
	// using postfix alignment attributes. Prefix works for alignment, but does not align
	// the size like postfix does.  Prefix also fails on templates.  So gcc style post fix
	// is still used, but the user will need to use TURBO_POSTFIX_ALIGN before the constructor parameters.
	#if defined(__GNUC__) && (__GNUC__ < 3)
		#define TURBO_ALIGN_OF(type) ((size_t)__alignof__(type))
		#define TURBO_ALIGN(n)
		#define TURBO_PREFIX_ALIGN(n)
		#define TURBO_POSTFIX_ALIGN(n) __attribute__((aligned(n)))
		#define TURBO_ALIGNED(variable_type, variable, n) variable_type variable __attribute__((aligned(n)))
		#define TURBO_PACKED __attribute__((packed))

	// GCC 3.x+, IBM, and clang support prefix attributes.
	#elif (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__xlC__) || defined(__clang__)
		#define TURBO_ALIGN_OF(type) ((size_t)__alignof__(type))
		#define TURBO_ALIGN(n) __attribute__((aligned(n)))
		#define TURBO_PREFIX_ALIGN(n)
		#define TURBO_POSTFIX_ALIGN(n) __attribute__((aligned(n)))
		#define TURBO_ALIGNED(variable_type, variable, n) variable_type variable __attribute__((aligned(n)))
		#define TURBO_PACKED __attribute__((packed))

	// Metrowerks supports prefix attributes.
	// Metrowerks does not support packed alignment attributes.
	#elif defined(TURBO_COMPILER_INTEL) || defined(CS_UNDEFINED_STRING) || (defined(TURBO_COMPILER_MSVC) && (TURBO_COMPILER_VERSION >= 1300))
		#define TURBO_ALIGN_OF(type) ((size_t)__alignof(type))
		#define TURBO_ALIGN(n) __declspec(align(n))
		#define TURBO_PREFIX_ALIGN(n) TURBO_ALIGN(n)
		#define TURBO_POSTFIX_ALIGN(n)
		#define TURBO_ALIGNED(variable_type, variable, n) TURBO_ALIGN(n) variable_type variable
		#define TURBO_PACKED // See TURBO_PRAGMA_PACK_VC for an alternative.

	// Arm brand compiler
	#elif defined(TURBO_COMPILER_ARM)
		#define TURBO_ALIGN_OF(type) ((size_t)__ALIGNOF__(type))
		#define TURBO_ALIGN(n) __align(n)
		#define TURBO_PREFIX_ALIGN(n) __align(n)
		#define TURBO_POSTFIX_ALIGN(n)
		#define TURBO_ALIGNED(variable_type, variable, n) __align(n) variable_type variable
		#define TURBO_PACKED __packed

	#else // Unusual compilers
		// There is nothing we can do about some of these. This is not as bad a problem as it seems.
		// If the given platform/compiler doesn't support alignment specifications, then it's somewhat
		// likely that alignment doesn't matter for that platform. Otherwise they would have defined
		// functionality to manipulate alignment.
		#define TURBO_ALIGN(n)
		#define TURBO_PREFIX_ALIGN(n)
		#define TURBO_POSTFIX_ALIGN(n)
		#define TURBO_ALIGNED(variable_type, variable, n) variable_type variable
		#define TURBO_PACKED

		#ifdef __cplusplus
			template <typename T> struct EAAlignOf1 { enum { s = sizeof (T), value = s ^ (s & (s - 1)) }; };
			template <typename T> struct EAAlignOf2;
			template <int size_diff> struct helper { template <typename T> struct Val { enum { value = size_diff }; }; };
			template <> struct helper<0> { template <typename T> struct Val { enum { value = EAAlignOf2<T>::value }; }; };
			template <typename T> struct EAAlignOf2 { struct Big { T x; char c; };
			enum { diff = sizeof (Big) - sizeof (T), value = helper<diff>::template Val<Big>::value }; };
			template <typename T> struct EAAlignof3 { enum { x = EAAlignOf2<T>::value, y = EAAlignOf1<T>::value, value = x < y ? x : y }; };
			#define TURBO_ALIGN_OF(type) ((size_t)EAAlignof3<type>::value)

		#else
			// C implementation of TURBO_ALIGN_OF
			// This implementation works for most cases, but doesn't directly work
			// for types such as function pointer declarations. To work with those
			// types you need to typedef the type and then use the typedef in TURBO_ALIGN_OF.
			#define TURBO_ALIGN_OF(type) ((size_t)offsetof(struct { char c; type m; }, m))
		#endif
	#endif

	// TURBO_PRAGMA_PACK_VC
	//
	// Wraps #pragma pack in a way that allows for cleaner code.
	//
	// Example usage:
	//    TURBO_PRAGMA_PACK_VC(push, 1)
	//    struct X{ char c; int i; };
	//    TURBO_PRAGMA_PACK_VC(pop)
	//
	#if !defined(TURBO_PRAGMA_PACK_VC)
		#if defined(TURBO_COMPILER_MSVC)
			#define TURBO_PRAGMA_PACK_VC(...) __pragma(pack(__VA_ARGS__))
		#elif !defined(TURBO_COMPILER_NO_VARIADIC_MACROS)
			#define TURBO_PRAGMA_PACK_VC(...)
		#else
			// No support. However, all compilers of significance to us support variadic macros.
		#endif
	#endif

	// ------------------------------------------------------------------------
	// TURBO_HAVE_INCLUDE_AVAILABLE
	//
	// Used to guard against the TURBO_HAVE_INCLUDE() macro on compilers that do not
	// support said feature.
	//
	// Example usage:
	//
	// #if TURBO_HAVE_INCLUDE_AVAILABLE
	//     #if TURBO_HAVE_INCLUDE("myinclude.h")
    //         #include "myinclude.h"
	//     #endif
	// #endif
	#if !defined(TURBO_HAVE_INCLUDE_AVAILABLE)
		#if defined(TURBO_COMPILER_CLANG) || defined(TURBO_COMPILER_GNUC)
			#define TURBO_HAVE_INCLUDE_AVAILABLE 1
		#else
			#define TURBO_HAVE_INCLUDE_AVAILABLE 0
		#endif
	#endif


	// ------------------------------------------------------------------------
	// TURBO_HAVE_INCLUDE
	//
	// May be used in #if and #elif expressions to test for the existence
	// of the header referenced in the operand. If possible it evaluates to a
	// non-zero value and zero otherwise. The operand is the same form as the file
	// in a #include directive.
	//
	// Example usage:
	//
	// #if TURBO_HAVE_INCLUDE("myinclude.h")
	//     #include "myinclude.h"
	// #endif
	//
	// #if TURBO_HAVE_INCLUDE(<myinclude.h>)
	//     #include <myinclude.h>
	// #endif

	#if !defined(TURBO_HAVE_INCLUDE)
		#if defined(TURBO_COMPILER_CLANG)
			#define TURBO_HAVE_INCLUDE(x) __has_include(x)
		#elif TURBO_COMPILER_GNUC
			#define TURBO_HAVE_INCLUDE(x) __has_include(x)
                #else
                        #define TURBO_HAVE_INCLUDE(x) 0
		#endif
	#endif


	// ------------------------------------------------------------------------
	// TURBO_INIT_PRIORITY_AVAILABLE
	//
	// This value is either not defined, or defined to 1.
	// Defines if the GCC attribute init_priority is supported by the compiler.
	//
	#if !defined(TURBO_INIT_PRIORITY_AVAILABLE)
		#if defined(__GNUC__) && !defined(__EDG__) // EDG typically #defines __GNUC__ but doesn't implement init_priority.
			#define TURBO_INIT_PRIORITY_AVAILABLE 1
		#elif defined(__clang__)
			#define TURBO_INIT_PRIORITY_AVAILABLE 1  // Clang implements init_priority
		#endif
	#endif


	// ------------------------------------------------------------------------
	// TURBO_INIT_PRIORITY
	//
	// This is simply a wrapper for the GCC init_priority attribute that allows
	// multiplatform code to be easier to read. This attribute doesn't apply
	// to VC++ because VC++ uses file-level pragmas to control init ordering.
	//
	// Example usage:
	//     SomeClass gSomeClass TURBO_INIT_PRIORITY(2000);
	//
	#if !defined(TURBO_INIT_PRIORITY)
		#if defined(TURBO_INIT_PRIORITY_AVAILABLE)
			#define TURBO_INIT_PRIORITY(x)  __attribute__ ((init_priority (x)))
		#else
			#define TURBO_INIT_PRIORITY(x)
		#endif
	#endif


	// ------------------------------------------------------------------------
	// TURBO_INIT_SEG_AVAILABLE
	//
	//
	#if !defined(TURBO_INIT_SEG_AVAILABLE)
		#if defined(_MSC_VER)
			#define TURBO_INIT_SEG_AVAILABLE 1
		#endif
	#endif


	// ------------------------------------------------------------------------
	// TURBO_INIT_SEG
	//
	// Specifies a keyword or code section that affects the order in which startup code is executed.
	//
	// https://docs.microsoft.com/en-us/cpp/preprocessor/init-seg?view=vs-2019
	//
	// Example:
	// 		TURBO_INIT_SEG(compiler) MyType gMyTypeGlobal;
	// 		TURBO_INIT_SEG("my_section") MyOtherType gMyOtherTypeGlobal;
	//
	#if !defined(TURBO_INIT_SEG)
		#if defined(TURBO_INIT_SEG_AVAILABLE)
			#define TURBO_INIT_SEG(x)                                                                                                \
				__pragma(warning(push)) __pragma(warning(disable : 4074)) __pragma(warning(disable : 4075)) __pragma(init_seg(x)) \
					__pragma(warning(pop))
		#else
			#define TURBO_INIT_SEG(x)
		#endif
	#endif


	// ------------------------------------------------------------------------
	// TURBO_MAY_ALIAS_AVAILABLE
	//
	// Defined as 0, 1, or 2.
	// Defines if the GCC attribute may_alias is supported by the compiler.
	// Consists of a value 0 (unsupported, shouldn't be used), 1 (some support),
	// or 2 (full proper support).
	//
	#ifndef TURBO_MAY_ALIAS_AVAILABLE
		#if defined(__GNUC__) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 303)
			#if   !defined(__EDG__)                 // define it as 1 while defining GCC's support as 2.
				#define TURBO_MAY_ALIAS_AVAILABLE 2
			#else
				#define TURBO_MAY_ALIAS_AVAILABLE 0
			#endif
		#else
			#define TURBO_MAY_ALIAS_AVAILABLE 0
		#endif
	#endif

	// ------------------------------------------------------------------------
	// TURBO_PASCAL
	//
	// Also known on PC platforms as stdcall.
	// This convention causes the compiler to assume that the called function
	// will pop off the stack space used to pass arguments, unless it takes a
	// variable number of arguments.
	//
	// Example usage:
	//    this:
	//       void DoNothing(int x);
	//       void DoNothing(int x){}
	//    would be written as this:
	//       void TURBO_PASCAL_FUNC(DoNothing(int x));
	//       void TURBO_PASCAL_FUNC(DoNothing(int x)){}
	//
	#ifndef TURBO_PASCAL
		#if defined(TURBO_COMPILER_MSVC)
			#define TURBO_PASCAL __stdcall
		#elif defined(TURBO_COMPILER_GNUC) && defined(TURBO_PROCESSOR_X86)
			#define TURBO_PASCAL __attribute__((stdcall))
		#else
			// Some compilers simply don't support pascal calling convention.
			// As a result, there isn't an issue here, since the specification of
			// pascal calling convention is for the purpose of disambiguating the
			// calling convention that is applied.
			#define TURBO_PASCAL
		#endif
	#endif

	#ifndef TURBO_PASCAL_FUNC
		#if defined(TURBO_COMPILER_MSVC)
			#define TURBO_PASCAL_FUNC(funcname_and_paramlist)    __stdcall funcname_and_paramlist
		#elif defined(TURBO_COMPILER_GNUC) && defined(TURBO_PROCESSOR_X86)
			#define TURBO_PASCAL_FUNC(funcname_and_paramlist)    __attribute__((stdcall)) funcname_and_paramlist
		#else
			#define TURBO_PASCAL_FUNC(funcname_and_paramlist)    funcname_and_paramlist
		#endif
	#endif

	// ------------------------------------------------------------------------
	// TURBO_ASAN_ENABLED
	//
	// Defined as 0 or 1. It's value depends on the compile environment.
	// Specifies whether the code is being built with Clang's Address Sanitizer.
	//
	#if defined(__has_feature)
		#if __has_feature(address_sanitizer)
			#define TURBO_ASAN_ENABLED 1
		#else
			#define TURBO_ASAN_ENABLED 0
		#endif
	#else
		#define TURBO_ASAN_ENABLED 0
	#endif


#endif  // TURBO_PLATFORM_CONFIG_COMPILER_TRAITS_H_










