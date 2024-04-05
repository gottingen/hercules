// Copyright 2024 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#pragma once

#include <collie/base/debug_assert.h>

#ifndef HERCULES_AST_ASSERTION_LEVEL
#    define HERCULES_AST_ASSERTION_LEVEL 0
#endif

#ifndef HERCULES_AST_PRECONDITION_LEVEL
#    ifdef NDEBUG
#        define HERCULES_AST_PRECONDITION_LEVEL 0
#    else
#        define HERCULES_AST_PRECONDITION_LEVEL 1
#    endif
#endif

namespace hercules::ccast::detail {
    struct assert_handler : collie::debug_assert::set_level<HERCULES_AST_ASSERTION_LEVEL>,
                            collie::debug_assert::default_handler {
    };

    struct precondition_error_handler : collie::debug_assert::set_level<HERCULES_AST_PRECONDITION_LEVEL>,
                                        collie::debug_assert::default_handler {
    };
} // namespace hercules::ccast::detail
