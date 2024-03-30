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

#include <hercules/ast/cc/cpp_expression.h>
#include <hercules/ast/cc/cpp_type.h>

namespace hercules::ccast {
    /// Additional base class for all [hercules::ccast::cpp_entity]() modelling some kind of variable.
    ///
    /// Examples are [hercules::ccast::cpp_variable]() or [hercules::ccast::cpp_function_parameter](),
    /// or anything that is name/type/default-value triple.
    class cpp_variable_base {
    public:
        /// \returns A reference to the [hercules::ccast::cpp_type]() of the variable.
        const cpp_type &type() const noexcept {
            return *type_;
        }

        /// \returns A [ts::optional_ref]() to the [hercules::ccast::cpp_expression]() that is the default value.
        collie::ts::optional_ref<const cpp_expression> default_value() const noexcept {
            return collie::ts::opt_ref(default_.get());
        }

    protected:
        cpp_variable_base(std::unique_ptr<cpp_type> type, std::unique_ptr<cpp_expression> def)
                : type_(std::move(type)), default_(std::move(def)) {}

        ~cpp_variable_base() noexcept = default;

    private:
        std::unique_ptr<cpp_type> type_;
        std::unique_ptr<cpp_expression> default_;
    };
} // namespace hercules::ccast
