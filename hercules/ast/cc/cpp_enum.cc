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


#include <hercules/ast/cc/cpp_enum.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

using namespace hercules::ccast;

cpp_entity_kind cpp_enum_value::kind() noexcept
{
    return cpp_entity_kind::enum_value_t;
}

std::unique_ptr<cpp_enum_value> cpp_enum_value::build(const cpp_entity_index& idx, cpp_entity_id id,
                                                      std::string                     name,
                                                      std::unique_ptr<cpp_expression> value)
{
    auto result
        = std::unique_ptr<cpp_enum_value>(new cpp_enum_value(std::move(name), std::move(value)));
    idx.register_definition(std::move(id), collie::ts::ref(*result));
    return result;
}

cpp_entity_kind cpp_enum_value::do_get_entity_kind() const noexcept
{
    return kind();
}

cpp_entity_kind cpp_enum::kind() noexcept
{
    return cpp_entity_kind::enum_t;
}

cpp_entity_kind cpp_enum::do_get_entity_kind() const noexcept
{
    return kind();
}

collie::ts::optional<cpp_scope_name> cpp_enum::do_get_scope_name() const
{
    if (scoped_)
        return collie::ts::ref(*this);
    return collie::ts::nullopt;
}
