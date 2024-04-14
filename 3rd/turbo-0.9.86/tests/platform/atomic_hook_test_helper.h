// Copyright 2020 The Turbo Authors.
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

#ifndef TURBO_PLATFORM_INTERNAL_ATOMIC_HOOK_TEST_HELPER_H_
#define TURBO_PLATFORM_INTERNAL_ATOMIC_HOOK_TEST_HELPER_H_

#include "turbo/platform/internal/atomic_hook.h"

namespace turbo {

namespace atomic_hook_internal {

using VoidF = void (*)();
extern turbo::base_internal::AtomicHook<VoidF> func;
extern int default_func_calls;
void DefaultFunc();
void RegisterFunc(VoidF func);

}  // namespace atomic_hook_internal

}  // namespace turbo

#endif  // TURBO_PLATFORM_INTERNAL_ATOMIC_HOOK_TEST_HELPER_H_
