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
#include <hercules/ast/cc/cpp_type.h>

namespace hercules::ccast
{
/// A [hercules::ccast::cpp_entity]() modelling a type alias/typedef.
/// \notes There is no distinction between `using` and `typedef` type aliases made in the AST.
class cpp_type_alias final : public cpp_entity
{
public:
    static cpp_entity_kind kind() noexcept;

    /// \returns A newly created and registered type alias.
    static std::unique_ptr<cpp_type_alias> build(const cpp_entity_index& idx, cpp_entity_id id,
                                                 std::string name, std::unique_ptr<cpp_type> type);

    /// \returns A newly created type alias that isn't registered.
    /// \notes This function is intendend for templated type aliases.
    static std::unique_ptr<cpp_type_alias> build(std::string name, std::unique_ptr<cpp_type> type);

    /// \returns A reference to the aliased [hercules::ccast::cpp_type]().
    const cpp_type& underlying_type() const noexcept
    {
        return *type_;
    }

private:
    cpp_type_alias(std::string name, std::unique_ptr<cpp_type> type)
    : cpp_entity(std::move(name)), type_(std::move(type))
    {}

    cpp_entity_kind do_get_entity_kind() const noexcept override;

    std::unique_ptr<cpp_type> type_;
};
} // namespace hercules::ccast

