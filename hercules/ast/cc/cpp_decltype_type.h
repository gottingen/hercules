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
    /// A [hercules::ccast::cpp_type]() that isn't given but taken from an expression.
    class cpp_decltype_type final : public cpp_type {
    public:
        /// \returns A newly created `decltype` type.
        static std::unique_ptr<cpp_decltype_type> build(std::unique_ptr<cpp_expression> expr) {
            return std::unique_ptr<cpp_decltype_type>(new cpp_decltype_type(std::move(expr)));
        }

        /// \returns A reference to the expression given.
        const cpp_expression &expression() const noexcept {
            return *expr_;
        }

    private:
        cpp_decltype_type(std::unique_ptr<cpp_expression> expr) : expr_(std::move(expr)) {}

        cpp_type_kind do_get_kind() const noexcept override {
            return cpp_type_kind::decltype_t;
        }

        std::unique_ptr<cpp_expression> expr_;
    };

    /// A [hercules::ccast::cpp_type]() that isn't given but deduced using the `decltype` rules.
    class cpp_decltype_auto_type final : public cpp_type {
    public:
        /// \returns A newly created `auto` type.
        static std::unique_ptr<cpp_decltype_auto_type> build() {
            return std::unique_ptr<cpp_decltype_auto_type>(new cpp_decltype_auto_type);
        }

    private:
        cpp_decltype_auto_type() = default;

        cpp_type_kind do_get_kind() const noexcept override {
            return cpp_type_kind::decltype_auto_t;
        }
    };
} // namespace hercules::ccast

