/*-----------------------------------------------------------------------------
 * config/compiler.h
 *
 * Copyright (c) Electronic Arts Inc. All rights reserved.
 *-----------------------------------------------------------------------------
 * Currently supported defines include:
 *     TURBO_COMPILER_GNUC
 *     TURBO_COMPILER_ARM
 *     TURBO_COMPILER_EDG
 *     TURBO_COMPILER_SN
 *     TURBO_COMPILER_MSVC
 *     TURBO_COMPILER_METROWERKS
 *     TURBO_COMPILER_INTEL
 *     TURBO_COMPILER_BORLANDC
 *     TURBO_COMPILER_IBM
 *     TURBO_COMPILER_QNX
 *     TURBO_COMPILER_GREEN_HILLS
 *     TURBO_COMPILER_CLANG
 *     TURBO_COMPILER_CLANG_CL
 *
 *     TURBO_COMPILER_VERSION = <integer>
 *     TURBO_COMPILER_NAME = <string>
 *     TURBO_COMPILER_STRING = <string>
 *
 *     TURBO_COMPILER_VA_COPY_REQUIRED
 *
 *  C++98/03 functionality
 *     TURBO_COMPILER_NO_STATIC_CONSTANTS
 *     TURBO_COMPILER_NO_TEMPLATE_SPECIALIZATION
 *     TURBO_COMPILER_NO_TEMPLATE_PARTIAL_SPECIALIZATION
 *     TURBO_COMPILER_NO_MEMBER_TEMPLATES
 *     TURBO_COMPILER_NO_MEMBER_TEMPLATE_SPECIALIZATION
 *     TURBO_COMPILER_NO_TEMPLATE_TEMPLATES
 *     TURBO_COMPILER_NO_MEMBER_TEMPLATE_FRIENDS
 *     TURBO_COMPILER_NO_VOID_RETURNS
 *     TURBO_COMPILER_NO_COVARIANT_RETURN_TYPE
 *     TURBO_COMPILER_NO_DEDUCED_TYPENAME
 *     TURBO_COMPILER_NO_ARGUMENT_DEPENDENT_LOOKUP
 *     TURBO_COMPILER_NO_EXCEPTION_STD_NAMESPACE
 *     TURBO_COMPILER_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS
 *     TURBO_COMPILER_NO_RTTI
 *     TURBO_COMPILER_NO_EXCEPTIONS
 *     TURBO_COMPILER_NO_NEW_THROW_SPEC
 *     TURBO_THROW_SPEC_NEW / TURBO_THROW_SPEC_DELETE
 *     TURBO_COMPILER_NO_UNWIND
 *     TURBO_COMPILER_NO_STANDARD_CPP_LIBRARY
 *     TURBO_COMPILER_NO_STATIC_VARIABLE_INIT
 *     TURBO_COMPILER_NO_STATIC_FUNCTION_INIT
 *     TURBO_COMPILER_NO_VARIADIC_MACROS
 *
 *  C++11 functionality
 *     TURBO_COMPILER_NO_RVALUE_REFERENCES
 *     TURBO_COMPILER_NO_RANGE_BASED_FOR_LOOP
 *     TURBO_COMPILER_NO_CONSTEXPR
 *     TURBO_COMPILER_NO_OVERRIDE
 *     TURBO_COMPILER_NO_INHERITANCE_FINAL
 *     TURBO_COMPILER_NO_NULLPTR
 *     TURBO_COMPILER_NO_AUTO
 *     TURBO_COMPILER_NO_DECLTYPE
 *     TURBO_COMPILER_NO_DEFAULTED_FUNCTIONS
 *     TURBO_COMPILER_NO_DELETED_FUNCTIONS
 *     TURBO_COMPILER_NO_LAMBDA_EXPRESSIONS
 *     TURBO_COMPILER_NO_TRAILING_RETURN_TYPES
 *     TURBO_COMPILER_NO_STRONGLY_TYPED_ENUMS
 *     TURBO_COMPILER_NO_FORWARD_DECLARED_ENUMS
 *     TURBO_COMPILER_NO_VARIADIC_TEMPLATES
 *     TURBO_COMPILER_NO_TEMPLATE_ALIASES
 *     TURBO_COMPILER_NO_INITIALIZER_LISTS
 *     TURBO_COMPILER_NO_NORETURN
 *     TURBO_COMPILER_NO_CARRIES_DEPENDENCY
 *     TURBO_COMPILER_NO_FALLTHROUGH
 *     TURBO_COMPILER_NO_NODISCARD
 *     TURBO_COMPILER_NO_MAYBE_UNUSED
 *     TURBO_COMPILER_NO_NONSTATIC_MEMBER_INITIALIZERS
 *     TURBO_COMPILER_NO_RIGHT_ANGLE_BRACKETS
 *     TURBO_COMPILER_NO_ALIGNOF
 *     TURBO_COMPILER_NO_ALIGNAS
 *     TURBO_COMPILER_NO_DELEGATING_CONSTRUCTORS
 *     TURBO_COMPILER_NO_INHERITING_CONSTRUCTORS
 *     TURBO_COMPILER_NO_USER_DEFINED_LITERALS
 *     TURBO_COMPILER_NO_STANDARD_LAYOUT_TYPES
 *     TURBO_COMPILER_NO_EXTENDED_SIZEOF
 *     TURBO_COMPILER_NO_INLINE_NAMESPACES
 *     TURBO_COMPILER_NO_UNRESTRICTED_UNIONS
 *     TURBO_COMPILER_NO_EXPLICIT_CONVERSION_OPERATORS
 *     TURBO_COMPILER_NO_FUNCTION_TEMPLATE_DEFAULT_ARGS
 *     TURBO_COMPILER_NO_LOCAL_CLASS_TEMPLATE_PARAMETERS
 *     TURBO_COMPILER_NO_NOEXCEPT
 *     TURBO_COMPILER_NO_RAW_LITERALS
 *     TURBO_COMPILER_NO_UNICODE_STRING_LITERALS
 *     TURBO_COMPILER_NO_NEW_CHARACTER_TYPES
 *     TURBO_COMPILER_NO_UNICODE_CHAR_NAME_LITERALS
 *     TURBO_COMPILER_NO_UNIFIED_INITIALIZATION_SYNTAX
 *     TURBO_COMPILER_NO_EXTENDED_FRIEND_DECLARATIONS
 *
 *  C++14 functionality
 *     TURBO_COMPILER_NO_VARIABLE_TEMPLATES
 *
 *  C++17 functionality
 *     TURBO_COMPILER_NO_INLINE_VARIABLES
 *     TURBO_COMPILER_NO_ALIGNED_NEW
 *
 *  C++20 functionality
 *     TURBO_COMPILER_NO_DESIGNATED_INITIALIZERS
 *
 *-----------------------------------------------------------------------------
 *
 * Supplemental documentation
 *     TURBO_COMPILER_NO_STATIC_CONSTANTS
 *         Code such as this is legal, but some compilers fail to compile it:
 *             struct A{ static const a = 1; };
 *
 *     TURBO_COMPILER_NO_TEMPLATE_SPECIALIZATION
 *         Some compilers fail to allow template specialization, such as with this:
 *             template<class U> void DoSomething(U u);
 *             void DoSomething(int x);
 *
 *     TURBO_COMPILER_NO_TEMPLATE_PARTIAL_SPECIALIZATION
 *         Some compilers fail to allow partial template specialization, such as with this:
 *             template <class T, class Allocator> class vector{ };         // Primary templated class.
 *             template <class Allocator> class vector<bool, Allocator>{ }; // Partially specialized version.
 *
 *     TURBO_COMPILER_NO_MEMBER_TEMPLATES
 *         Some compilers fail to allow member template functions such as this:
 *             struct A{ template<class U> void DoSomething(U u); };
 *
 *     TURBO_COMPILER_NO_MEMBER_TEMPLATE_SPECIALIZATION
 *         Some compilers fail to allow member template specialization, such as with this:
 *             struct A{
 *                 template<class U> void DoSomething(U u);
 *                 void DoSomething(int x);
 *             };
 *
 *     TURBO_COMPILER_NO_TEMPLATE_TEMPLATES
 *         Code such as this is legal:
 *             template<typename T, template<typename> class U>
 *             U<T> SomeFunction(const U<T> x) { return x.DoSomething(); }
 *
 *     TURBO_COMPILER_NO_MEMBER_TEMPLATE_FRIENDS
 *         Some compilers fail to compile templated friends, as with this:
 *             struct A{ template<class U> friend class SomeFriend; };
 *         This is described in the C++ Standard at 14.5.3.
 *
 *     TURBO_COMPILER_NO_VOID_RETURNS
 *          This is legal C++:
 *              void DoNothing1(){ };
 *              void DoNothing2(){ return DoNothing1(); }
 *
 *     TURBO_COMPILER_NO_COVARIANT_RETURN_TYPE
 *         See the C++ standard sec 10.3,p5.
 *
 *     TURBO_COMPILER_NO_DEDUCED_TYPENAME
 *         Some compilers don't support the use of 'typename' for
 *         dependent types in deduced contexts, as with this:
 *             template <class T> void Function(T, typename T::type);
 *
 *     TURBO_COMPILER_NO_ARGUMENT_DEPENDENT_LOOKUP
 *         Also known as Koenig lookup. Basically, if you have a function
 *         that is a namespace and you call that function without prefixing
 *         it with the namespace the compiler should look at any arguments
 *         you pass to that function call and search their namespace *first*
 *         to see if the given function exists there.
 *
 *     TURBO_COMPILER_NO_EXCEPTION_STD_NAMESPACE
 *         <exception> is in namespace std. Some std libraries fail to
 *         put the contents of <exception> in namespace std. The following
 *         code should normally be legal:
 *             void Function(){ std::terminate(); }
 *
 *     TURBO_COMPILER_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS
 *         Some compilers fail to execute DoSomething() properly, though they
 *         succeed in compiling it, as with this:
 *             template <int i>
 *             bool DoSomething(int j){ return i == j; };
 *             DoSomething<1>(2);
 *
 *     TURBO_COMPILER_NO_EXCEPTIONS
 *         The compiler is configured to disallow the use of try/throw/catch
 *         syntax (often to improve performance). Use of such syntax in this
 *         case will cause a compilation error.
 *
 *     TURBO_COMPILER_NO_UNWIND
 *         The compiler is configured to allow the use of try/throw/catch
 *         syntax and behaviour but disables the generation of stack unwinding
 *         code for responding to exceptions (often to improve performance).
 *
 *---------------------------------------------------------------------------*/

#ifndef TURBO_PLATFORM_CONFIG_COMPILER_H_
#define TURBO_PLATFORM_CONFIG_COMPILER_H_

	#include "turbo/platform/config/platform.h"

	// Note: This is used to generate the TURBO_COMPILER_STRING macros
	#ifndef INTERNAL_STRINGIZE
		#define INTERNAL_STRINGIZE(x) INTERNAL_PRIMITIVE_STRINGIZE(x)
	#endif
	#ifndef INTERNAL_PRIMITIVE_STRINGIZE
		#define INTERNAL_PRIMITIVE_STRINGIZE(x) #x
	#endif


	// EDG (EDG compiler front-end, used by other compilers such as SN)
	#if defined(__EDG_VERSION__)
		#define TURBO_COMPILER_EDG 1

		#if defined(_MSC_VER)
			#define TURBO_COMPILER_EDG_VC_MODE 1
		#endif
		#if defined(__GNUC__)
			#define TURBO_COMPILER_EDG_GCC_MODE 1
		#endif
	#endif

	// TURBO_COMPILER_WINRTCX_ENABLED
	//
	// Defined as 1 if the compiler has its available C++/CX support enabled, else undefined.
	// This specifically means the corresponding compilation unit has been built with Windows Runtime
	// Components enabled, usually via the '-ZW' compiler flags being used. This option allows for using
	// ref counted hat-type '^' objects and other C++/CX specific keywords like "ref new"
	#if !defined(TURBO_COMPILER_WINRTCX_ENABLED) && defined(__cplusplus_winrt)
		#define TURBO_COMPILER_WINRTCX_ENABLED 1
	#endif


	// TURBO_COMPILER_CPP11_ENABLED
	//
	// Defined as 1 if the compiler has its available C++11 support enabled, else undefined.
	// This does not mean that all of C++11 or any particular feature of C++11 is supported
	// by the compiler. It means that whatever C++11 support the compiler has is enabled.
	// This also includes existing and older compilers that still identify C++11 as C++0x.
	//
	// We cannot use (__cplusplus >= 201103L) alone because some compiler vendors have
	// decided to not define __cplusplus like thus until they have fully completed their
	// C++11 support.
	//
	#if !defined(TURBO_COMPILER_CPP11_ENABLED) && defined(__cplusplus)
		#if (__cplusplus >= 201103L)    // Clang and GCC defines this like so in C++11 mode.
			#define TURBO_COMPILER_CPP11_ENABLED 1
		#elif defined(__GNUC__) && defined(__GXX_EXPERIMENTAL_CXX0X__)
			#define TURBO_COMPILER_CPP11_ENABLED 1
		#elif defined(_MSC_VER) && _MSC_VER >= 1600         // Microsoft unilaterally enables its C++11 support; there is no way to disable it.
			#define TURBO_COMPILER_CPP11_ENABLED 1
		#elif defined(__EDG_VERSION__) // && ???
			// To do: Is there a generic way to determine this?
		#endif
	#endif


	// TURBO_COMPILER_CPP20_ENABLED
	//
	// Defined as 1 if the compiler has its available C++20 support enabled, else undefined.
	// This does not mean that all of C++20 or any particular feature of C++20 is supported
	// by the compiler. It means that whatever C++20 support the compiler has is enabled.
 	//
	// We cannot use (__cplusplus >= 202003L) alone because some compiler vendors have
	// decided to not define __cplusplus like thus until they have fully completed their
	// C++20 support.
	#if !defined(TURBO_COMPILER_CPP20_ENABLED) && defined(__cplusplus)
 		// TODO(rparoin): enable once a C++20 value for the __cplusplus macro has been published
		// #if (__cplusplus >= 202003L)
		//     #define TURBO_COMPILER_CPP20_ENABLED 1
		// #elif defined(_MSVC_LANG) && (_MSVC_LANG >= 202003L) // C++20+
		//     #define TURBO_COMPILER_CPP20_ENABLED 1
		// #endif
	#endif



	#if   defined(__ARMCC_VERSION)
		// Note that this refers to the ARM RVCT compiler (armcc or armcpp), but there
		// are other compilers that target ARM processors, such as GCC and Microsoft VC++.
		// If you want to detect compiling for the ARM processor, check for TURBO_PROCESSOR_ARM
		// being defined.
		// This compiler is also identified by defined(__CC_ARM) || defined(__ARMCC__).
		#define TURBO_COMPILER_RVCT    1
		#define TURBO_COMPILER_ARM     1
		#define TURBO_COMPILER_VERSION __ARMCC_VERSION
		#define TURBO_COMPILER_NAME    "RVCT"
	  //#define TURBO_COMPILER_STRING (defined below)

	// Clang's GCC-compatible driver.
	#elif defined(__clang__) && !defined(_MSC_VER)
		#define TURBO_COMPILER_CLANG   1
		#define TURBO_COMPILER_VERSION (__clang_major__ * 100 + __clang_minor__)
		#define TURBO_COMPILER_NAME    "clang"
		#define TURBO_COMPILER_STRING  TURBO_COMPILER_NAME __clang_version__

	// GCC (a.k.a. GNUC)
	#elif defined(__GNUC__) // GCC compilers exist for many platforms.
		#define TURBO_COMPILER_GNUC    1
		#define TURBO_COMPILER_VERSION (__GNUC__ * 1000 + __GNUC_MINOR__)
		#define TURBO_COMPILER_NAME    "GCC"
		#define TURBO_COMPILER_STRING  TURBO_COMPILER_NAME " compiler, version " INTERNAL_STRINGIZE( __GNUC__ ) "." INTERNAL_STRINGIZE( __GNUC_MINOR__ )

		#if (__GNUC__ == 2) && (__GNUC_MINOR__ < 95) // If GCC < 2.95...
			#define TURBO_COMPILER_NO_MEMBER_TEMPLATES 1
		#endif
		#if (__GNUC__ == 2) && (__GNUC_MINOR__ <= 97) // If GCC <= 2.97...
			#define TURBO_COMPILER_NO_MEMBER_TEMPLATE_FRIENDS 1
		#endif
		#if (__GNUC__ == 3) && ((__GNUC_MINOR__ == 1) || (__GNUC_MINOR__ == 2)) // If GCC 3.1 or 3.2 (but not pre 3.1 or post 3.2)...
			#define TURBO_COMPILER_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS 1
		#endif

	// Borland C++
	#elif defined(__BORLANDC__)
		#define TURBO_COMPILER_BORLANDC 1
		#define TURBO_COMPILER_VERSION  __BORLANDC__
		#define TURBO_COMPILER_NAME     "Borland C"
	  //#define TURBO_COMPILER_STRING (defined below)

		#if (__BORLANDC__ <= 0x0550)      // If Borland C++ Builder 4 and 5...
			#define TURBO_COMPILER_NO_MEMBER_TEMPLATE_FRIENDS 1
		#endif
		#if (__BORLANDC__ >= 0x561) && (__BORLANDC__ < 0x600)
			#define TURBO_COMPILER_NO_MEMBER_FUNCTION_SPECIALIZATION 1
		#endif


	// Intel C++
	// The Intel Windows compiler masquerades as VC++ and defines _MSC_VER.
	// The Intel compiler is based on the EDG compiler front-end.
	#elif defined(__ICL) || defined(__ICC)
		#define TURBO_COMPILER_INTEL 1

		// Should we enable the following? We probably should do so since enabling it does a lot more good than harm
		// for users. The Intel Windows compiler does a pretty good job of emulating VC++ and so the user would likely
		// have to handle few special cases where the Intel compiler doesn't emulate VC++ correctly.
		#if defined(_MSC_VER)
			#define TURBO_COMPILER_MSVC 1
			#define TURBO_COMPILER_MICROSOFT 1
		#endif

		// Should we enable the following? This isn't as clear because as of this writing we don't know if the Intel
		// compiler truly emulates GCC well enough that enabling this does more good than harm.
		#if defined(__GNUC__)
			#define TURBO_COMPILER_GNUC 1
		#endif

		#if defined(__ICL)
			#define TURBO_COMPILER_VERSION __ICL
		#elif defined(__ICC)
			#define TURBO_COMPILER_VERSION __ICC
		#endif
		#define TURBO_COMPILER_NAME "Intel C++"
		#if defined(_MSC_VER)
			#define TURBO_COMPILER_STRING  TURBO_COMPILER_NAME " compiler, version " INTERNAL_STRINGIZE( TURBO_COMPILER_VERSION ) ", EDG version " INTERNAL_STRINGIZE( __EDG_VERSION__ ) ", VC++ version " INTERNAL_STRINGIZE( _MSC_VER )
		#elif defined(__GNUC__)
			#define TURBO_COMPILER_STRING  TURBO_COMPILER_NAME " compiler, version " INTERNAL_STRINGIZE( TURBO_COMPILER_VERSION ) ", EDG version " INTERNAL_STRINGIZE( __EDG_VERSION__ ) ", GCC version " INTERNAL_STRINGIZE( __GNUC__ )
		#else
			#define TURBO_COMPILER_STRING  TURBO_COMPILER_NAME " compiler, version " INTERNAL_STRINGIZE( TURBO_COMPILER_VERSION ) ", EDG version " INTERNAL_STRINGIZE( __EDG_VERSION__ )
		#endif


	#elif defined(_MSC_VER)
		#define TURBO_COMPILER_MSVC 1
		#define TURBO_COMPILER_MICROSOFT 1
		#define TURBO_COMPILER_VERSION _MSC_VER
		#define TURBO_COMPILER_NAME "Microsoft Visual C++"
	  //#define TURBO_COMPILER_STRING (defined below)

		#if defined(__clang__)
			// Clang's MSVC-compatible driver.
			#define TURBO_COMPILER_CLANG_CL 1
		#endif

		#define TURBO_STANDARD_LIBRARY_MSVC 1
		#define TURBO_STANDARD_LIBRARY_MICROSOFT 1

		#if (_MSC_VER <= 1200) // If VC6.x and earlier...
			#if (_MSC_VER < 1200)
				#define TURBO_COMPILER_MSVCOLD 1
			#else
				#define TURBO_COMPILER_MSVC6 1
			#endif

			#if (_MSC_VER < 1200) // If VC5.x or earlier...
				#define TURBO_COMPILER_NO_TEMPLATE_SPECIALIZATION 1
			#endif
			#define TURBO_COMPILER_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS 1     // The compiler compiles this OK, but executes it wrong. Fixed in VC7.0
			#define TURBO_COMPILER_NO_VOID_RETURNS 1                             // The compiler fails to compile such cases. Fixed in VC7.0
			#define TURBO_COMPILER_NO_EXCEPTION_STD_NAMESPACE 1                  // The compiler fails to compile such cases. Fixed in VC7.0
			#define TURBO_COMPILER_NO_DEDUCED_TYPENAME 1                         // The compiler fails to compile such cases. Fixed in VC7.0
			#define TURBO_COMPILER_NO_STATIC_CONSTANTS 1                         // The compiler fails to compile such cases. Fixed in VC7.0
			#define TURBO_COMPILER_NO_COVARIANT_RETURN_TYPE 1                    // The compiler fails to compile such cases. Fixed in VC7.1
			#define TURBO_COMPILER_NO_ARGUMENT_DEPENDENT_LOOKUP 1                // The compiler compiles this OK, but executes it wrong. Fixed in VC7.1
			#define TURBO_COMPILER_NO_TEMPLATE_TEMPLATES 1                       // The compiler fails to compile such cases. Fixed in VC7.1
			#define TURBO_COMPILER_NO_TEMPLATE_PARTIAL_SPECIALIZATION 1          // The compiler fails to compile such cases. Fixed in VC7.1
			#define TURBO_COMPILER_NO_MEMBER_TEMPLATE_FRIENDS 1                  // The compiler fails to compile such cases. Fixed in VC7.1
			//#define TURBO_COMPILER_NO_MEMBER_TEMPLATES 1                       // VC6.x supports member templates properly 95% of the time. So do we flag the remaining 5%?
			//#define TURBO_COMPILER_NO_MEMBER_TEMPLATE_SPECIALIZATION 1         // VC6.x supports member templates properly 95% of the time. So do we flag the remaining 5%?

		#elif (_MSC_VER <= 1300) // If VC7.0 and earlier...
			#define TURBO_COMPILER_MSVC7 1

			#define TURBO_COMPILER_NO_COVARIANT_RETURN_TYPE 1                    // The compiler fails to compile such cases. Fixed in VC7.1
			#define TURBO_COMPILER_NO_ARGUMENT_DEPENDENT_LOOKUP 1                // The compiler compiles this OK, but executes it wrong. Fixed in VC7.1
			#define TURBO_COMPILER_NO_TEMPLATE_TEMPLATES 1                       // The compiler fails to compile such cases. Fixed in VC7.1
			#define TURBO_COMPILER_NO_TEMPLATE_PARTIAL_SPECIALIZATION 1          // The compiler fails to compile such cases. Fixed in VC7.1
			#define TURBO_COMPILER_NO_MEMBER_TEMPLATE_FRIENDS 1                  // The compiler fails to compile such cases. Fixed in VC7.1
			#define TURBO_COMPILER_NO_MEMBER_FUNCTION_SPECIALIZATION 1           // This is the case only for VC7.0 and not VC6 or VC7.1+. Fixed in VC7.1
			//#define TURBO_COMPILER_NO_MEMBER_TEMPLATES 1                       // VC7.0 supports member templates properly 95% of the time. So do we flag the remaining 5%?

		#elif (_MSC_VER < 1400) // VS2003       _MSC_VER of 1300 means VC7 (VS2003)
			// The VC7.1 and later compiler is fairly close to the C++ standard
			// and thus has no compiler limitations that we are concerned about.
			#define TURBO_COMPILER_MSVC7_2003 1
			#define TURBO_COMPILER_MSVC7_1    1

		#elif (_MSC_VER < 1500) // VS2005       _MSC_VER of 1400 means VC8 (VS2005)
			#define TURBO_COMPILER_MSVC8_2005 1
			#define TURBO_COMPILER_MSVC8_0    1

		#elif (_MSC_VER < 1600) // VS2008.      _MSC_VER of 1500 means VC9 (VS2008)
			#define TURBO_COMPILER_MSVC9_2008 1
			#define TURBO_COMPILER_MSVC9_0    1

		#elif (_MSC_VER < 1700) // VS2010       _MSC_VER of 1600 means VC10 (VS2010)
			#define TURBO_COMPILER_MSVC_2010 1
			#define TURBO_COMPILER_MSVC10_0  1

		#elif (_MSC_VER < 1800) // VS2012       _MSC_VER of 1700 means VS2011/VS2012
			#define TURBO_COMPILER_MSVC_2011 1   // Microsoft changed the name to VS2012 before shipping, despite referring to it as VS2011 up to just a few weeks before shipping.
			#define TURBO_COMPILER_MSVC11_0  1
			#define TURBO_COMPILER_MSVC_2012 1
			#define TURBO_COMPILER_MSVC12_0  1

		#elif (_MSC_VER < 1900) // VS2013       _MSC_VER of 1800 means VS2013
			#define TURBO_COMPILER_MSVC_2013 1
			#define TURBO_COMPILER_MSVC13_0  1

		#elif (_MSC_VER < 1910) // VS2015       _MSC_VER of 1900 means VS2015
			#define TURBO_COMPILER_MSVC_2015 1
			#define TURBO_COMPILER_MSVC14_0  1

		#elif (_MSC_VER < 1911) // VS2017       _MSC_VER of 1910 means VS2017
			#define TURBO_COMPILER_MSVC_2017 1
			#define TURBO_COMPILER_MSVC15_0  1

		#endif


	// IBM
	#elif defined(__xlC__)
		#define TURBO_COMPILER_IBM     1
		#define TURBO_COMPILER_NAME    "IBM XL C"
		#define TURBO_COMPILER_VERSION __xlC__
		#define TURBO_COMPILER_STRING "IBM XL C compiler, version " INTERNAL_STRINGIZE( __xlC__ )

	// Unknown
	#else // Else the compiler is unknown

		#define TURBO_COMPILER_VERSION 0
		#define TURBO_COMPILER_NAME   "Unknown"

	#endif

	#ifndef TURBO_COMPILER_STRING
		#define TURBO_COMPILER_STRING TURBO_COMPILER_NAME " compiler, version " INTERNAL_STRINGIZE(TURBO_COMPILER_VERSION)
	#endif

	// TURBO_COMPILER_NO_DESIGNATED_INITIALIZERS
	//
	// Indicates the target compiler supports the C++20 "designated initializer" language feature.
	// https://en.cppreference.com/w/cpp/language/aggregate_initialization
	//
	// Example:
	//   struct A { int x; int y; };
	//   A a = { .y = 42, .x = 1 };
	//
	#if !defined(TURBO_COMPILER_NO_DESIGNATED_INITIALIZERS)
		#if defined(TURBO_COMPILER_CPP20_ENABLED)
			// supported.
		#else
			#define TURBO_COMPILER_NO_DESIGNATED_INITIALIZERS 1
		#endif
	#endif


        // Portable check for GCC minimum version:
        // https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
        #if defined(__GNUC__) && defined(__GNUC_MINOR__)
        #define TURBO_HAVE_MIN_GNUC_VERSION(x, y) \
          (__GNUC__ > (x) || __GNUC__ == (x) && __GNUC_MINOR__ >= (y))
        #else
        #define TURBO_HAVE_MIN_GNUC_VERSION(x, y) 0
        #endif

        #if defined(__clang__) && defined(__clang_major__) && defined(__clang_minor__)
        #define TURBO_HAVE_MIN_CLANG_VERSION(x, y) \
          (__clang_major__ > (x) || __clang_major__ == (x) && __clang_minor__ >= (y))
        #else
        #define TURBO_HAVE_MIN_CLANG_VERSION(x, y) 0
        #endif

        #ifndef TURBO_GCC_VERSION
            #if defined(TURBO_COMPILER_GNUC)
                #define TURBO_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
            #else
                #define TURBO_GCC_VERSION 0
            #endif
        #endif  // TURBO_GCC_VERSION

        #ifndef TURBO_CLANG_VERSION
            #if defined(TURBO_COMPILER_CLANG)
                #define TURBO_CLANG_VERSION (__clang_major__ * 100 + __clang_minor__)
            #else
                #define TURBO_CLANG_VERSION 0
            #endif
        #endif  // TURBO_CLANG_VERSION

        #ifdef __ICL
        #  define TURBO_ICC_VERSION __ICL
        #elif defined(__INTEL_COMPILER)
        #  define TURBO_ICC_VERSION __INTEL_COMPILER
        #else
        #  define TURBO_ICC_VERSION 0
        #endif

        #ifdef _MSC_VER
        #  define TURBO_MSC_VERSION _MSC_VER
        #  define TURBO_MSC_WARNING(...) __pragma(warning(__VA_ARGS__))
        #else
        #  define TURBO_MSC_VERSION 0
        #  define TURBO_MSC_WARNING(...)
        #endif

        #ifndef TURBO_GCC_PRAGMA
        // Workaround _Pragma bug https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59884.
        #  if TURBO_GCC_VERSION >= 504
        #    define TURBO_GCC_PRAGMA(arg) _Pragma(arg)
        #  else
        #    define TURBO_GCC_PRAGMA(arg)
        #  endif
        #endif

        #ifdef __NVCC__
        #  define TURBO_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
        #else
        #  define TURBO_CUDA_VERSION 0
        #endif


#endif // TURBO_PLATFORM_CONFIG_COMPILER_H_





