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


#include <hercules/ast/cc/cpp_entity_index.h>

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_entity_kind.h>
#include <hercules/ast/cc/cpp_file.h>
#include <hercules/ast/cc/detail/assert.h>

using namespace hercules::ccast;

cpp_entity_index::duplicate_definition_error::duplicate_definition_error()
: std::logic_error("duplicate registration of entity definition")
{}

void cpp_entity_index::register_definition(cpp_entity_id                           id,
                                           collie::ts::object_ref<const cpp_entity> entity) const
{
    DEBUG_ASSERT(entity->kind() != cpp_entity_kind::namespace_t,
                 detail::precondition_error_handler{}, "must not be a namespace");
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        result = map_.emplace(std::move(id), value(entity, true));
    if (!result.second)
    {
        // already in map, override declaration
        auto& value = result.first->second;
        if (value.is_definition && !is_template(value.entity->kind()) && entity->parent()
            && !is_template(entity->parent().value().kind()))
            // allow duplicate definition of templates
            // this handles things such as SFINAE
            throw duplicate_definition_error();
        value.is_definition = true;
        value.entity        = entity;
    }
}

bool cpp_entity_index::register_file(cpp_entity_id                         id,
                                     collie::ts::object_ref<const cpp_file> file) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return map_.emplace(std::move(id), value(file, true)).second;
}

void cpp_entity_index::register_forward_declaration(
    cpp_entity_id id, collie::ts::object_ref<const cpp_entity> entity) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    map_.emplace(std::move(id), value(entity, false));
}

void cpp_entity_index::register_namespace(cpp_entity_id                              id,
                                          collie::ts::object_ref<const cpp_namespace> ns) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    ns_[std::move(id)].push_back(ns);
}

collie::ts::optional_ref<const cpp_entity> cpp_entity_index::lookup(
    const cpp_entity_id& id) const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        iter = map_.find(id);
    if (iter == map_.end())
        return {};
    return collie::ts::ref(iter->second.entity.get());
}

collie::ts::optional_ref<const cpp_entity> cpp_entity_index::lookup_definition(
    const cpp_entity_id& id) const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        iter = map_.find(id);
    if (iter == map_.end() || !iter->second.is_definition)
        return {};
    return collie::ts::ref(iter->second.entity.get());
}

auto cpp_entity_index::lookup_namespace(const cpp_entity_id& id) const noexcept
    -> collie::ts::array_ref<collie::ts::object_ref<const cpp_namespace>>
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        iter = ns_.find(id);
    if (iter == ns_.end())
        return nullptr;
    auto& vec = iter->second;
    return collie::ts::ref(vec.data(), vec.size());
}
