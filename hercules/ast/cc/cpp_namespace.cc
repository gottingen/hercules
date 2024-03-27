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


#include <hercules/ast/cc/cpp_namespace.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

using namespace hercules::ccast;

cpp_entity_kind cpp_namespace::kind() noexcept
{
    return cpp_entity_kind::namespace_t;
}

cpp_entity_kind cpp_namespace::do_get_entity_kind() const noexcept
{
    return kind();
}

bool detail::cpp_namespace_ref_predicate::operator()(const cpp_entity& e)
{
    return e.kind() == cpp_entity_kind::namespace_t;
}

std::unique_ptr<cpp_namespace_alias> cpp_namespace_alias::build(const cpp_entity_index& idx,
                                                                cpp_entity_id id, std::string name,
                                                                cpp_namespace_ref target)
{
    auto ptr = std::unique_ptr<cpp_namespace_alias>(
        new cpp_namespace_alias(std::move(name), std::move(target)));
    idx.register_forward_declaration(std::move(id), collie::ts::ref(*ptr));
    return ptr;
}

cpp_entity_kind cpp_namespace_alias::kind() noexcept
{
    return cpp_entity_kind::namespace_alias_t;
}

cpp_entity_kind cpp_namespace_alias::do_get_entity_kind() const noexcept
{
    return kind();
}

cpp_entity_kind cpp_using_directive::kind() noexcept
{
    return cpp_entity_kind::using_directive_t;
}

cpp_entity_kind cpp_using_directive::do_get_entity_kind() const noexcept
{
    return kind();
}

cpp_entity_kind cpp_using_declaration::kind() noexcept
{
    return cpp_entity_kind::using_declaration_t;
}

cpp_entity_kind cpp_using_declaration::do_get_entity_kind() const noexcept
{
    return kind();
}
