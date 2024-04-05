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
    /// An array of a [hercules::ccast::cpp_type]().
    class cpp_array_type final : public cpp_type {
    public:
        /// \returns A newly created array.
        /// \notes `size` may be `nullptr`.
        static std::unique_ptr<cpp_array_type> build(std::unique_ptr<cpp_type> type,
                                                     std::unique_ptr<cpp_expression> size) {
            return std::unique_ptr<cpp_array_type>(
                    new cpp_array_type(std::move(type), std::move(size)));
        }

        /// \returns A reference to the value [hercules::ccast::cpp_type]().
        const cpp_type &value_type() const noexcept {
            return *type_;
        }

        /// \returns An optional reference to the [hercules::ccast::cpp_expression]() that is the size of the
        /// array. \notes An unsized array - `T[]` - does not have a size.
        collie::ts::optional_ref<const cpp_expression> size() const noexcept {
            return collie::ts::opt_cref(size_.get());
        }

    private:
        cpp_array_type(std::unique_ptr<cpp_type> type, std::unique_ptr<cpp_expression> size)
                : type_(std::move(type)), size_(std::move(size)) {}

        cpp_type_kind do_get_kind() const noexcept override {
            return cpp_type_kind::array_t;
        }

        std::unique_ptr<cpp_type> type_;
        std::unique_ptr<cpp_expression> size_;
    };
} // namespace hercules::ccast
