// Copyright 2023 The titan-search Authors.
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

#ifndef TURBO_FLAGS_DECLARE_H_
#define TURBO_FLAGS_DECLARE_H_

#include "turbo/platform/port.h"

namespace turbo {
namespace flags_internal {

// turbo::Flag<T> represents a flag of type 'T' created by TURBO_FLAG.
template <typename T>
class Flag;

}  // namespace flags_internal

// Flag
//
// Forward declaration of the `turbo::Flag` type for use in defining the macro.
#if defined(_MSC_VER) && !defined(__clang__)
template <typename T>
class Flag;
#else
template <typename T>
using Flag = flags_internal::Flag<T>;
#endif

}  // namespace turbo

// TURBO_DECLARE_FLAG()
//
// This macro is a convenience for declaring use of an `turbo::Flag` within a
// translation unit. This macro should be used within a header file to
// declare usage of the flag within any .cc file including that header file.
//
// The TURBO_DECLARE_FLAG(type, name) macro expands to:
//
//   extern turbo::Flag<type> FLAGS_name;
#define TURBO_DECLARE_FLAG(type, name) TURBO_DECLARE_FLAG_INTERNAL(type, name)

// Internal implementation of TURBO_DECLARE_FLAG to allow macro expansion of its
// arguments. Clients must use TURBO_DECLARE_FLAG instead.
#define TURBO_DECLARE_FLAG_INTERNAL(type, name)               \
  extern turbo::Flag<type> FLAGS_##name;                      \
  namespace turbo /* block flags in namespaces */ {}          \
  /* second redeclaration is to allow applying attributes */ \
  extern turbo::Flag<type> FLAGS_##name

#endif  // TURBO_FLAGS_DECLARE_H_
