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

#include "atomic_hook_test_helper.h"

#include "turbo/platform/port.h"
#include "turbo/platform/internal/atomic_hook.h"

namespace turbo {

    namespace atomic_hook_internal {

        TURBO_INTERNAL_ATOMIC_HOOK_ATTRIBUTES turbo::base_internal::AtomicHook<VoidF>
                func(DefaultFunc);
        TURBO_CONST_INIT int default_func_calls = 0;

        void DefaultFunc() { default_func_calls++; }

        void RegisterFunc(VoidF f) { func.Store(f); }

    }  // namespace atomic_hook_internal

}  // namespace turbo
