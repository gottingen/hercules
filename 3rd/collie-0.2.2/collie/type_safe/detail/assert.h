// Copyright 2024 The Elastic-AI Authors.
// part of Elastic AI Search
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
//

#pragma once

#include <collie/base/debug_assert.h>
#include <collie/type_safe/config.h>

namespace collie::ts::detail {

    struct assert_handler : collie::debug_assert::set_level<TYPE_SAFE_ENABLE_ASSERTIONS>,
                            collie::debug_assert::default_handler {
    };

    struct precondition_error_handler
            : collie::debug_assert::set_level<TYPE_SAFE_ENABLE_PRECONDITION_CHECKS>,
              collie::debug_assert::default_handler {
    };

    inline void on_disabled_exception() noexcept {
        struct handler : collie::debug_assert::set_level<1>, collie::debug_assert::default_handler {
        };
        DEBUG_UNREACHABLE(handler{}, "attempt to throw an exception but exceptions are disabled");
    }
} // namespace collie::ts::detail
