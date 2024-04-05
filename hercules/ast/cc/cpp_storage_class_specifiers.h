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

#include <hercules/ast/cc/cppast_fwd.h>

namespace hercules::ccast {
    /// C++ storage class specifiers.
    ///
    /// See http://en.cppreference.com/w/cpp/language/storage_duration, for example.
    /// \notes These are just all the possible *keywords* used in a variable declaration,
    /// not necessarily their *semantic* meaning.
    enum cpp_storage_class_specifiers : int {
        cpp_storage_class_none = 0, //< no storage class specifier given.

        cpp_storage_class_auto = 1, //< *automatic* storage duration.

        cpp_storage_class_static = 2, //< *static* or *thread* storage duration and *internal* linkage.
        cpp_storage_class_extern = 4, //< *static* or *thread* storage duration and *external* linkage.

        cpp_storage_class_thread_local = 8, //< *thread* storage duration.
        /// \notes This is the only one that can be combined with the others.
    };

    /// \returns Whether the [hercules::ccast::cpp_storage_class_specifiers]() contain `thread_local`.
    inline bool is_thread_local(cpp_storage_class_specifiers spec) noexcept {
        return (spec & cpp_storage_class_thread_local) != 0;
    }

    /// \returns Whether the [hercules::ccast::cpp_storage_class_speicifers]() contain `auto`.
    inline bool is_auto(cpp_storage_class_specifiers spec) noexcept {
        return (spec & cpp_storage_class_auto) != 0;
    }

    /// \returns Whether the [hercules::ccast::cpp_storage_class_specifiers]() contain `static`.
    inline bool is_static(cpp_storage_class_specifiers spec) noexcept {
        return (spec & cpp_storage_class_static) != 0;
    }

    /// \returns Whether the [hercules::ccast::cpp_storage_class_specifiers]() contain `extern`.
    inline bool is_extern(cpp_storage_class_specifiers spec) noexcept {
        return (spec & cpp_storage_class_extern) != 0;
    }
} // namespace hercules::ccast
