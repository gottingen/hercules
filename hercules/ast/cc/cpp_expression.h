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

#include <atomic>
#include <memory>

#include <hercules/ast/cc/cpp_token.h>
#include <hercules/ast/cc/cpp_type.h>

namespace hercules::ccast
{
/// The kind of a [hercules::ccast::cpp_expression]().
enum class cpp_expression_kind
{
    literal_t,

    unexposed_t,
};

/// Base class for all C++ expressions.
class cpp_expression
{
public:
    cpp_expression(const cpp_expression&)            = delete;
    cpp_expression& operator=(const cpp_expression&) = delete;

    virtual ~cpp_expression() noexcept = default;

    /// \returns The [hercules::ccast::cpp_expression_kind]().
    cpp_expression_kind kind() const noexcept
    {
        return do_get_kind();
    }

    /// \returns The type of the expression.
    const cpp_type& type() const noexcept
    {
        return *type_;
    }

    /// \returns The specified user data.
    void* user_data() const noexcept
    {
        return user_data_.load();
    }

    /// \effects Sets some kind of user data.
    ///
    /// User data is just some kind of pointer, there are no requirements.
    /// The class will do no lifetime management.
    ///
    /// User data is useful if you need to store additional data for an entity without the need to
    /// maintain a registry.
    void set_user_data(void* data) const noexcept
    {
        user_data_ = data;
    }

protected:
    /// \effects Creates it given the type.
    /// \requires The type must not be `nullptr`.
    cpp_expression(std::unique_ptr<cpp_type> type) : type_(std::move(type)), user_data_(nullptr)
    {
        DEBUG_ASSERT(type_ != nullptr, detail::precondition_error_handler{});
    }

private:
    /// \returns The [hercules::ccast::cpp_expression_kind]().
    virtual cpp_expression_kind do_get_kind() const noexcept = 0;

    std::unique_ptr<cpp_type>  type_;
    mutable std::atomic<void*> user_data_;
};

/// An unexposed [hercules::ccast::cpp_expression]().
///
/// There is no further information than a string available.
class cpp_unexposed_expression final : public cpp_expression
{
public:
    /// \returns A newly created unexposed expression.
    static std::unique_ptr<cpp_unexposed_expression> build(std::unique_ptr<cpp_type> type,
                                                           cpp_token_string          str)
    {
        return std::unique_ptr<cpp_unexposed_expression>(
            new cpp_unexposed_expression(std::move(type), std::move(str)));
    }

    /// \returns The expression as a string.
    const cpp_token_string& expression() const noexcept
    {
        return str_;
    }

private:
    cpp_unexposed_expression(std::unique_ptr<cpp_type> type, cpp_token_string str)
    : cpp_expression(std::move(type)), str_(std::move(str))
    {}

    cpp_expression_kind do_get_kind() const noexcept override
    {
        return cpp_expression_kind::unexposed_t;
    }

    cpp_token_string str_;
};

/// A [hercules::ccast::cpp_expression]() that is a literal.
class cpp_literal_expression final : public cpp_expression
{
public:
    /// \returns A newly created literal expression.
    static std::unique_ptr<cpp_literal_expression> build(std::unique_ptr<cpp_type> type,
                                                         std::string               value)
    {
        return std::unique_ptr<cpp_literal_expression>(
            new cpp_literal_expression(std::move(type), std::move(value)));
    }

    /// \returns The value of the literal, as string.
    const std::string& value() const noexcept
    {
        return value_;
    }

private:
    cpp_literal_expression(std::unique_ptr<cpp_type> type, std::string value)
    : cpp_expression(std::move(type)), value_(std::move(value))
    {}

    cpp_expression_kind do_get_kind() const noexcept override
    {
        return cpp_expression_kind::literal_t;
    }

    std::string value_;
};

/// \exclude
namespace detail
{
    void write_expression(code_generator::output& output, const cpp_expression& expr);
} // namespace detail
} // namespace hercules::ccast
