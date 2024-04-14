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

#ifndef TURBO_PLATFORM_CONFIG_ATTRIBUTE_LITERAL_H_
#define TURBO_PLATFORM_CONFIG_ATTRIBUTE_LITERAL_H_

#include "turbo/platform/config/compiler_traits.h"

// ------------------------------------------------------------------------
// TURBO_CONCAT
//
// This macro joins the two arguments together, even when one of
// the arguments is itself a macro (see 16.3.1 in C++98 standard).
// This is often used to create a unique name with __LINE__.
//
// For example, this declaration:
//    char TURBO_CONCAT(unique_, __LINE__);
// expands to this:
//    char unique_73;
//
// Note that all versions of MSVC++ up to at least version 7.1
// fail to properly compile macros that use __LINE__ in them
// when the "program database for edit and continue" option
// is enabled. The result is that __LINE__ gets converted to
// something like __LINE__(Var+37).
//
#ifndef TURBO_CONCAT
#define TURBO_CONCAT(a, b)  TURBO_CONCAT1(a, b)
#define TURBO_CONCAT1(a, b) TURBO_CONCAT2(a, b)
#define TURBO_CONCAT2(a, b) a##b
#endif

// TURBO_PRETTY_FUNCTION
//
// In C++11, __func__ gives the undecorated name of the current function.  That
// is, "main", not "int main()".  Various compilers give extra macros to get the
// decorated function name, including return type and arguments, to
// differentiate between overload sets.  TURBO_PRETTY_FUNCTION is a portable
// version of these macros which forwards to the correct macro on each compiler.
#if defined(_MSC_VER)
#define TURBO_PRETTY_FUNCTION __FUNCSIG__
#elif defined(__GNUC__)
#define TURBO_PRETTY_FUNCTION __PRETTY_FUNCTION__
#else
#error "Unsupported compiler"
#endif

// ------------------------------------------------------------------------
// TURBO_STRINGIFY
//
// Example usage:
//     printf("Line: %s", TURBO_STRINGIFY(__LINE__));
//
#ifndef TURBO_STRINGIFY
#define TURBO_STRINGIFY(x)     TURBO_STRINGIFYIMPL(x)
#define TURBO_STRINGIFYIMPL(x) #x
#endif


// ------------------------------------------------------------------------
// TURBO_IDENTITY
//
#ifndef TURBO_IDENTITY
#define TURBO_IDENTITY(x) x
#endif


#endif  // TURBO_PLATFORM_CONFIG_ATTRIBUTE_LITERAL_H_
