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

#ifndef TURBO_PLATFORM_CONFIG_WARN_CONFIG_H_
#define TURBO_PLATFORM_CONFIG_WARN_CONFIG_H_
#include "turbo/platform/config/compiler_traits.h"

// ------------------------------------------------------------------------
// TURBO_DISABLE_VC_WARNING / TURBO_RESTORE_VC_WARNING
//
// Disable and re-enable warning(s) within code.
// This is simply a wrapper for VC++ #pragma warning(disable: nnnn) for the
// purpose of making code easier to read due to avoiding nested compiler ifdefs
// directly in code.
//
// Example usage:
//     TURBO_DISABLE_VC_WARNING(4127 3244)
//     <code>
//     TURBO_RESTORE_VC_WARNING()
//
#ifndef TURBO_DISABLE_VC_WARNING
#if defined(_MSC_VER)
#define TURBO_DISABLE_VC_WARNING(w)  \
				__pragma(warning(push))       \
				__pragma(warning(disable:w))
#else
#define TURBO_DISABLE_VC_WARNING(w)
#endif
#endif

#ifndef TURBO_RESTORE_VC_WARNING
#if defined(_MSC_VER)
#define TURBO_RESTORE_VC_WARNING()   \
				__pragma(warning(pop))
#else
#define TURBO_RESTORE_VC_WARNING()
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_DISABLE_ALL_GCC_WARNINGS / TURBO_RESTORE_ALL_GCC_WARNINGS
//
// This isn't possible except via using _Pragma("GCC system_header"), though
// that has some limitations in how it works. Another means is to manually
// disable individual warnings within a GCC diagnostic push statement.
// GCC doesn't have as many warnings as VC++ and EDG and so this may be feasible.
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// TURBO_ENABLE_GCC_WARNING_AS_ERROR / TURBO_DISABLE_GCC_WARNING_AS_ERROR
//
// Example usage:
//     // Only one warning can be treated as an error per statement, due to how GCC works.
//     TURBO_ENABLE_GCC_WARNING_AS_ERROR(-Wuninitialized)
//     TURBO_ENABLE_GCC_WARNING_AS_ERROR(-Wunused)
//     <code>
//     TURBO_DISABLE_GCC_WARNING_AS_ERROR()
//     TURBO_DISABLE_GCC_WARNING_AS_ERROR()
//
#ifndef TURBO_ENABLE_GCC_WARNING_AS_ERROR
#if defined(TURBO_COMPILER_GNUC)
#define EAGCCWERRORHELP0(x) #x
#define EAGCCWERRORHELP1(x) EAGCCWERRORHELP0(GCC diagnostic error x)
#define EAGCCWERRORHELP2(x) EAGCCWERRORHELP1(#x)
#endif

#if defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION >= 4006) // Can't test directly for __GNUC__ because some compilers lie.
#define TURBO_ENABLE_GCC_WARNING_AS_ERROR(w)   \
				_Pragma("GCC diagnostic push")  \
				_Pragma(EAGCCWERRORHELP2(w))
#elif defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION >= 4004)
#define TURBO_DISABLE_GCC_WARNING(w)   \
				_Pragma(EAGCCWERRORHELP2(w))
#else
#define TURBO_DISABLE_GCC_WARNING(w)
#endif
#endif

#ifndef TURBO_DISABLE_GCC_WARNING_AS_ERROR
#if defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION >= 4006)
#define TURBO_DISABLE_GCC_WARNING_AS_ERROR()    \
				_Pragma("GCC diagnostic pop")
#else
#define TURBO_DISABLE_GCC_WARNING_AS_ERROR()
#endif
#endif


// ------------------------------------------------------------------------
// TURBO_ENABLE_VC_WARNING_AS_ERROR / TURBO_DISABLE_VC_WARNING_AS_ERROR
//
// Disable and re-enable treating a warning as error within code.
// This is simply a wrapper for VC++ #pragma warning(error: nnnn) for the
// purpose of making code easier to read due to avoiding nested compiler ifdefs
// directly in code.
//
// Example usage:
//     TURBO_ENABLE_VC_WARNING_AS_ERROR(4996)
//     <code>
//     TURBO_DISABLE_VC_WARNING_AS_ERROR()
//
#ifndef TURBO_ENABLE_VC_WARNING_AS_ERROR
#if defined(_MSC_VER)
#define TURBO_ENABLE_VC_WARNING_AS_ERROR(w) \
					__pragma(warning(push)) \
					__pragma(warning(error:w))
#else
#define TURBO_ENABLE_VC_WARNING_AS_ERROR(w)
#endif
#endif

#ifndef TURBO_DISABLE_VC_WARNING_AS_ERROR
#if defined(_MSC_VER)
#define TURBO_DISABLE_VC_WARNING_AS_ERROR() \
				__pragma(warning(pop))
#else
#define TURBO_DISABLE_VC_WARNING_AS_ERROR()
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_DISABLE_ALL_VC_WARNINGS / TURBO_RESTORE_ALL_VC_WARNINGS
//
// Disable and re-enable all warning(s) within code.
//
// Example usage:
//     TURBO_DISABLE_ALL_VC_WARNINGS()
//     <code>
//     TURBO_RESTORE_ALL_VC_WARNINGS()
//
//This is duplicated from TBBase's eacompilertraits.h
#ifndef TURBO_DISABLE_ALL_VC_WARNINGS
#if defined(_MSC_VER)
#define TURBO_DISABLE_ALL_VC_WARNINGS()  \
				__pragma(warning(push, 0)) \
				__pragma(warning(disable: 4244 4265 4267 4350 4472 4509 4548 4623 4710 4985 6320 4755 4625 4626 4702)) // Some warnings need to be explicitly called out.
#else
#define TURBO_DISABLE_ALL_VC_WARNINGS()
#endif
#endif

#ifndef TURBO_RESTORE_ALL_VC_WARNINGS
#if defined(_MSC_VER)
#define TURBO_RESTORE_ALL_VC_WARNINGS()  \
				__pragma(warning(pop))
#else
#define TURBO_RESTORE_ALL_VC_WARNINGS()
#endif
#endif



// ------------------------------------------------------------------------
// TURBO_DISABLE_GCC_WARNING / TURBO_RESTORE_GCC_WARNING
//
// Example usage:
//     // Only one warning can be ignored per statement, due to how GCC works.
//     TURBO_DISABLE_GCC_WARNING(-Wuninitialized)
//     TURBO_DISABLE_GCC_WARNING(-Wunused)
//     <code>
//     TURBO_RESTORE_GCC_WARNING()
//     TURBO_RESTORE_GCC_WARNING()
//
#ifndef TURBO_DISABLE_GCC_WARNING
#if defined(TURBO_COMPILER_GNUC)
#define EAGCCWHELP0(x) #x
#define EAGCCWHELP1(x) EAGCCWHELP0(GCC diagnostic ignored x)
#define EAGCCWHELP2(x) EAGCCWHELP1(#x)
#endif

#if defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION >= 4006) // Can't test directly for __GNUC__ because some compilers lie.
#define TURBO_DISABLE_GCC_WARNING(w)   \
				_Pragma("GCC diagnostic push")  \
				_Pragma(EAGCCWHELP2(w))
#elif defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION >= 4004)
#define TURBO_DISABLE_GCC_WARNING(w)   \
				_Pragma(EAGCCWHELP2(w))
#else
#define TURBO_DISABLE_GCC_WARNING(w)
#endif
#endif

#ifndef TURBO_RESTORE_GCC_WARNING
#if defined(TURBO_COMPILER_GNUC) && (TURBO_COMPILER_VERSION >= 4006)
#define TURBO_RESTORE_GCC_WARNING()    \
				_Pragma("GCC diagnostic pop")
#else
#define TURBO_RESTORE_GCC_WARNING()
#endif
#endif


// ------------------------------------------------------------------------
// TURBO_DISABLE_CLANG_WARNING / TURBO_RESTORE_CLANG_WARNING
//
// Example usage:
//     // Only one warning can be ignored per statement, due to how clang works.
//     TURBO_DISABLE_CLANG_WARNING(-Wuninitialized)
//     TURBO_DISABLE_CLANG_WARNING(-Wunused)
//     <code>
//     TURBO_RESTORE_CLANG_WARNING()
//     TURBO_RESTORE_CLANG_WARNING()
//
#ifndef TURBO_DISABLE_CLANG_WARNING
#if defined(TURBO_COMPILER_CLANG) || defined(TURBO_COMPILER_CLANG_CL)
#define EACLANGWHELP0(x) #x
#define EACLANGWHELP1(x) EACLANGWHELP0(clang diagnostic ignored x)
#define EACLANGWHELP2(x) EACLANGWHELP1(#x)

#define TURBO_DISABLE_CLANG_WARNING(w)   \
				_Pragma("clang diagnostic push")  \
				_Pragma(EACLANGWHELP2(-Wunknown-warning-option))\
				_Pragma(EACLANGWHELP2(w))
#else
#define TURBO_DISABLE_CLANG_WARNING(w)
#endif
#endif

#ifndef TURBO_RESTORE_CLANG_WARNING
#if defined(TURBO_COMPILER_CLANG) || defined(TURBO_COMPILER_CLANG_CL)
#define TURBO_RESTORE_CLANG_WARNING()    \
				_Pragma("clang diagnostic pop")
#else
#define TURBO_RESTORE_CLANG_WARNING()
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_DISABLE_ALL_CLANG_WARNINGS / TURBO_RESTORE_ALL_CLANG_WARNINGS
//
// The situation for clang is the same as for GCC. See above.
// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// TURBO_ENABLE_CLANG_WARNING_AS_ERROR / TURBO_DISABLE_CLANG_WARNING_AS_ERROR
//
// Example usage:
//     // Only one warning can be treated as an error per statement, due to how clang works.
//     TURBO_ENABLE_CLANG_WARNING_AS_ERROR(-Wuninitialized)
//     TURBO_ENABLE_CLANG_WARNING_AS_ERROR(-Wunused)
//     <code>
//     TURBO_DISABLE_CLANG_WARNING_AS_ERROR()
//     TURBO_DISABLE_CLANG_WARNING_AS_ERROR()
//
#ifndef TURBO_ENABLE_CLANG_WARNING_AS_ERROR
#if defined(TURBO_COMPILER_CLANG) || defined(TURBO_COMPILER_CLANG_CL)
#define EACLANGWERRORHELP0(x) #x
#define EACLANGWERRORHELP1(x) EACLANGWERRORHELP0(clang diagnostic error x)
#define EACLANGWERRORHELP2(x) EACLANGWERRORHELP1(#x)

#define TURBO_ENABLE_CLANG_WARNING_AS_ERROR(w)   \
				_Pragma("clang diagnostic push")  \
				_Pragma(EACLANGWERRORHELP2(w))
#else
#define TURBO_DISABLE_CLANG_WARNING(w)
#endif
#endif

#ifndef TURBO_DISABLE_CLANG_WARNING_AS_ERROR
#if defined(TURBO_COMPILER_CLANG) || defined(TURBO_COMPILER_CLANG_CL)
#define TURBO_DISABLE_CLANG_WARNING_AS_ERROR()    \
				_Pragma("clang diagnostic pop")
#else
#define TURBO_DISABLE_CLANG_WARNING_AS_ERROR()
#endif
#endif

// ------------------------------------------------------------------------
// TURBO_DISABLE_SN_WARNING / TURBO_RESTORE_SN_WARNING
//
// Note that we define this macro specifically for the SN compiler instead of
// having a generic one for EDG-based compilers. The reason for this is that
// while SN is indeed based on EDG, SN has different warning value mappings
// and thus warning 1234 for SN is not the same as 1234 for all other EDG compilers.
//
// Example usage:
//     // Currently we are limited to one warning per line.
//     TURBO_DISABLE_SN_WARNING(1787)
//     TURBO_DISABLE_SN_WARNING(552)
//     <code>
//     TURBO_RESTORE_SN_WARNING()
//     TURBO_RESTORE_SN_WARNING()
//
#ifndef TURBO_DISABLE_SN_WARNING
#define TURBO_DISABLE_SN_WARNING(w)
#endif

#ifndef TURBO_RESTORE_SN_WARNING
#define TURBO_RESTORE_SN_WARNING()
#endif


// ------------------------------------------------------------------------
// TURBO_DISABLE_ALL_SN_WARNINGS / TURBO_RESTORE_ALL_SN_WARNINGS
//
// Example usage:
//     TURBO_DISABLE_ALL_SN_WARNINGS()
//     <code>
//     TURBO_RESTORE_ALL_SN_WARNINGS()
//
#ifndef TURBO_DISABLE_ALL_SN_WARNINGS
#define TURBO_DISABLE_ALL_SN_WARNINGS()
#endif

#ifndef TURBO_RESTORE_ALL_SN_WARNINGS
#define TURBO_RESTORE_ALL_SN_WARNINGS()
#endif



// ------------------------------------------------------------------------
// TURBO_DISABLE_GHS_WARNING / TURBO_RESTORE_GHS_WARNING
//
// Disable warnings from the Green Hills compiler.
//
// Example usage:
//     TURBO_DISABLE_GHS_WARNING(193)
//     TURBO_DISABLE_GHS_WARNING(236, 5323)
//     <code>
//     TURBO_RESTORE_GHS_WARNING()
//     TURBO_RESTORE_GHS_WARNING()
//
#ifndef TURBO_DISABLE_GHS_WARNING
#define TURBO_DISABLE_GHS_WARNING(w)
#endif

#ifndef TURBO_RESTORE_GHS_WARNING
#define TURBO_RESTORE_GHS_WARNING()
#endif


// ------------------------------------------------------------------------
// TURBO_DISABLE_ALL_GHS_WARNINGS / TURBO_RESTORE_ALL_GHS_WARNINGS
//
// #ifndef TURBO_DISABLE_ALL_GHS_WARNINGS
//     #if defined(TURBO_COMPILER_GREEN_HILLS)
//         #define TURBO_DISABLE_ALL_GHS_WARNINGS(w)  \_
//             _Pragma("_________")
//     #else
//         #define TURBO_DISABLE_ALL_GHS_WARNINGS(w)
//     #endif
// #endif
//
// #ifndef TURBO_RESTORE_ALL_GHS_WARNINGS
//     #if defined(TURBO_COMPILER_GREEN_HILLS)
//         #define TURBO_RESTORE_ALL_GHS_WARNINGS()   \_
//             _Pragma("_________")
//     #else
//         #define TURBO_RESTORE_ALL_GHS_WARNINGS()
//     #endif
// #endif



// ------------------------------------------------------------------------
// TURBO_DISABLE_EDG_WARNING / TURBO_RESTORE_EDG_WARNING
//
// Example usage:
//     // Currently we are limited to one warning per line.
//     TURBO_DISABLE_EDG_WARNING(193)
//     TURBO_DISABLE_EDG_WARNING(236)
//     <code>
//     TURBO_RESTORE_EDG_WARNING()
//     TURBO_RESTORE_EDG_WARNING()
//
#ifndef TURBO_DISABLE_EDG_WARNING
// EDG-based compilers are inconsistent in how the implement warning pragmas.
#if defined(TURBO_COMPILER_EDG) && !defined(TURBO_COMPILER_INTEL) && !defined(TURBO_COMPILER_RVCT)
#define EAEDGWHELP0(x) #x
#define EAEDGWHELP1(x) EAEDGWHELP0(diag_suppress x)

#define TURBO_DISABLE_EDG_WARNING(w)   \
				_Pragma("control %push diag")   \
				_Pragma(EAEDGWHELP1(w))
#else
#define TURBO_DISABLE_EDG_WARNING(w)
#endif
#endif

#ifndef TURBO_RESTORE_EDG_WARNING
#if defined(TURBO_COMPILER_EDG) && !defined(TURBO_COMPILER_INTEL) && !defined(TURBO_COMPILER_RVCT)
#define TURBO_RESTORE_EDG_WARNING()   \
				_Pragma("control %pop diag")
#else
#define TURBO_RESTORE_EDG_WARNING()
#endif
#endif


// ------------------------------------------------------------------------
// TURBO_DISABLE_ALL_EDG_WARNINGS / TURBO_RESTORE_ALL_EDG_WARNINGS
//
//#ifndef TURBO_DISABLE_ALL_EDG_WARNINGS
//    #if defined(TURBO_COMPILER_EDG) && !defined(TURBO_COMPILER_SN)
//        #define TURBO_DISABLE_ALL_EDG_WARNINGS(w)  \_
//            _Pragma("_________")
//    #else
//        #define TURBO_DISABLE_ALL_EDG_WARNINGS(w)
//    #endif
//#endif
//
//#ifndef TURBO_RESTORE_ALL_EDG_WARNINGS
//    #if defined(TURBO_COMPILER_EDG) && !defined(TURBO_COMPILER_SN)
//        #define TURBO_RESTORE_ALL_EDG_WARNINGS()   \_
//            _Pragma("_________")
//    #else
//        #define TURBO_RESTORE_ALL_EDG_WARNINGS()
//    #endif
//#endif



// ------------------------------------------------------------------------
// TURBO_DISABLE_CW_WARNING / TURBO_RESTORE_CW_WARNING
//
// Note that this macro can only control warnings via numbers and not by
// names. The reason for this is that the compiler's syntax for such
// warnings is not the same as for numbers.
//
// Example usage:
//     // Currently we are limited to one warning per line and must also specify the warning in the restore macro.
//     TURBO_DISABLE_CW_WARNING(10317)
//     TURBO_DISABLE_CW_WARNING(10324)
//     <code>
//     TURBO_RESTORE_CW_WARNING(10317)
//     TURBO_RESTORE_CW_WARNING(10324)
//
#ifndef TURBO_DISABLE_CW_WARNING
#define TURBO_DISABLE_CW_WARNING(w)
#endif

#ifndef TURBO_RESTORE_CW_WARNING

#define TURBO_RESTORE_CW_WARNING(w)

#endif


// ------------------------------------------------------------------------
// TURBO_DISABLE_ALL_CW_WARNINGS / TURBO_RESTORE_ALL_CW_WARNINGS
//
#ifndef TURBO_DISABLE_ALL_CW_WARNINGS
#define TURBO_DISABLE_ALL_CW_WARNINGS()

#endif

#ifndef TURBO_RESTORE_ALL_CW_WARNINGS
#define TURBO_RESTORE_ALL_CW_WARNINGS()
#endif



#endif // TURBO_PLATFORM_CONFIG_WARN_CONFIG_H_
