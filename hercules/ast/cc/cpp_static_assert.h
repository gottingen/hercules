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

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_expression.h>

namespace hercules::ccast
{
class cpp_static_assert : public cpp_entity
{
public:
    static cpp_entity_kind kind() noexcept;

    /// \returns A newly created `static_assert()` entity.
    /// \notes It will not be registered as nothing can refer to it.
    static std::unique_ptr<cpp_static_assert> build(std::unique_ptr<cpp_expression> expr,
                                                    std::string                     msg)
    {
        return std::unique_ptr<cpp_static_assert>(
            new cpp_static_assert(std::move(expr), std::move(msg)));
    }

    /// \returns A reference to the [hercules::ccast::cpp_expression]() that is being asserted.
    const cpp_expression& expression() const noexcept
    {
        return *expr_;
    }

    /// \returns A reference to the message of the assertion.
    const std::string& message() const noexcept
    {
        return msg_;
    }

private:
    cpp_static_assert(std::unique_ptr<cpp_expression> expr, std::string msg)
    : cpp_entity(""), expr_(std::move(expr)), msg_(std::move(msg))
    {}

    cpp_entity_kind do_get_entity_kind() const noexcept override;

    std::unique_ptr<cpp_expression> expr_;
    std::string                     msg_;
};
} // namespace hercules::ccast
