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

#ifndef TURBO_PLATFORM_CONFIG_ATTRIBUTE_TEMPLATE_H_
#define TURBO_PLATFORM_CONFIG_ATTRIBUTE_TEMPLATE_H_

#include "turbo/platform/config/compiler_traits.h"

#ifndef TURBO_USE_USER_DEFINED_LITERALS
// EDG based compilers (Intel, NVIDIA, Elbrus, etc), GCC and MSVC support UDLs.
#if TURBO_HAVE_FEATURE(cxx_user_literals) || \
              (defined(TURBO_COMPILER_GNUC) && TURBO_COMPILER_VERSION >= 4007) || \
             ((defined(TURBO_COMPILER_MSVC) &&  TURBO_COMPILER_VERSION >= 1900) && (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= /* UDL feature */ 480))
#define TURBO_USE_USER_DEFINED_LITERALS 1
#else
#define TURBO_USE_USER_DEFINED_LITERALS 0
#endif
#endif

#endif // TURBO_PLATFORM_CONFIG_ATTRIBUTE_TEMPLATE_H_
