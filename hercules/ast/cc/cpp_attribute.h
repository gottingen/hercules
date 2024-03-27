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

#include <string>
#include <vector>

#include <collie/type_safe/optional.h>
#include <collie/type_safe/optional_ref.h>

#include <hercules/ast/cc/cpp_token.h>

namespace hercules::ccast
{
/// The known C++ attributes.
enum class cpp_attribute_kind
{
    // update get_attribute_kind() in tokenizer, when updating this

    alignas_,
    carries_dependency,
    deprecated,
    fallthrough,
    maybe_unused,
    nodiscard,
    noreturn,

    unknown, //< An unknown attribute.
};

/// A C++ attribute, including `alignas` specifiers.
///
/// It consists of a name, an optional namespace scope and optional arguments.
/// The scope is just a single identifier and doesn't include the `::` and can be given explicitly
/// or via using. The arguments are as specified in the source code but do not include the
/// outer-most `(` and `)`. It can also be variadic or not.
///
/// An attribute can be known or unknown.
/// A known attribute will have the [hercules::ccast::cpp_attribute_kind]() set properly.
class cpp_attribute
{
public:
    /// \effects Creates a known attribute, potentially with arguments.
    cpp_attribute(cpp_attribute_kind kind, collie::ts::optional<cpp_token_string> arguments);

    /// \effects Creates an unknown attribute giving it the optional scope, names, arguments and
    /// whether it is variadic.
    cpp_attribute(collie::ts::optional<std::string> scope, std::string name,
                  collie::ts::optional<cpp_token_string> arguments, bool is_variadic)
    : scope_(std::move(scope)), arguments_(std::move(arguments)), name_(std::move(name)),
      variadic_(is_variadic)
    {}

    /// \returns The kind of attribute, if it is known.
    const cpp_attribute_kind& kind() const noexcept
    {
        return kind_;
    }

    /// \returns The name of the attribute.
    const std::string& name() const noexcept
    {
        return name_;
    }

    /// \returns The scope of the attribute, if there is any.
    const collie::ts::optional<std::string>& scope() const noexcept
    {
        return scope_;
    }

    /// \returns Whether or not the attribute is variadic.
    bool is_variadic() const noexcept
    {
        return variadic_;
    }

    /// \returns The arguments of the attribute, if they are any.
    const collie::ts::optional<cpp_token_string>& arguments() const noexcept
    {
        return arguments_;
    }

private:
    collie::ts::optional<std::string>      scope_;
    collie::ts::optional<cpp_token_string> arguments_;
    std::string                           name_;
    cpp_attribute_kind                    kind_ = cpp_attribute_kind::unknown;
    bool                                  variadic_;
};

/// A list of C++ attributes.
using cpp_attribute_list = std::vector<cpp_attribute>;

/// Checks whether an attribute is given.
/// \returns `true` if the given attribute list (1-2) / entity (3-4) contain
/// an attribute of the given name (1+3) / kind (2+4).
/// `false` otherwise.
/// \group has_attribute
collie::ts::optional_ref<const cpp_attribute> has_attribute(const cpp_attribute_list& attributes,
                                                           const std::string&        name);

/// \group has_attribute
collie::ts::optional_ref<const cpp_attribute> has_attribute(const cpp_attribute_list& attributes,
                                                           cpp_attribute_kind        kind);

/// \group has_attribute
collie::ts::optional_ref<const cpp_attribute> has_attribute(const cpp_entity&  e,
                                                           const std::string& name);

/// \group has_attribute
collie::ts::optional_ref<const cpp_attribute> has_attribute(const cpp_entity&  e,
                                                           cpp_attribute_kind kind);
} // namespace hercules::ccast
