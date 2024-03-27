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

#include <vector>

#include <collie/type_safe/variant.h>

#include <hercules/ast/cc/cpp_entity_index.h>
#include <hercules/ast/cc/detail/assert.h>

namespace hercules::ccast
{

/// A basic reference to some kind of [hercules::ccast::cpp_entity]().
///
/// It can either refer to a single [hercules::ccast::cpp_entity]()
/// or multiple.
/// In the later case it is *overloaded*.
template <typename T, typename Predicate>
class basic_cpp_entity_ref
{
public:
    /// \effects Creates it giving it the target id and name.
    basic_cpp_entity_ref(cpp_entity_id target_id, std::string target_name)
    : target_(std::move(target_id)), name_(std::move(target_name))
    {}

    /// \effects Creates it giving it multiple target ids and name.
    /// \notes This is to refer to an overloaded function.
    basic_cpp_entity_ref(std::vector<cpp_entity_id> target_ids, std::string target_name)
    : target_(std::move(target_ids)), name_(std::move(target_name))
    {}

    /// \returns The name of the reference, as spelled in the source code.
    const std::string& name() const noexcept
    {
        return name_;
    }

    /// \returns Whether or not it refers to multiple entities.
    bool is_overloaded() const noexcept
    {
        return target_.has_value(collie::ts::variant_type<std::vector<cpp_entity_id>>{});
    }

    /// \returns The number of entities it refers to.
    collie::ts::size_t no_overloaded() const noexcept
    {
        return id().size();
    }

    /// \returns An array reference to the id or ids it refers to.
    collie::ts::array_ref<const cpp_entity_id> id() const noexcept
    {
        if (is_overloaded())
        {
            auto& vec = target_.value(collie::ts::variant_type<std::vector<cpp_entity_id>>{});
            return collie::ts::ref(vec.data(), vec.size());
        }
        else
        {
            auto& id = target_.value(collie::ts::variant_type<cpp_entity_id>{});
            return collie::ts::ref(&id, 1u);
        }
    }

    /// \returns An array reference to the entities it refers to.
    /// The return type provides `operator[]` + `size()`,
    /// as well as `begin()` and `end()` returning forward iterators.
    /// \exclude return
    std::vector<collie::ts::object_ref<const T>> get(const cpp_entity_index& idx) const
    {
        std::vector<collie::ts::object_ref<const T>> result;
        get_impl(std::is_convertible<cpp_namespace&, T&>{}, result, idx);
        return result;
    }

private:
    void get_impl(std::true_type, std::vector<collie::ts::object_ref<const T>>& result,
                  const cpp_entity_index& idx) const
    {
        for (auto& cur : id())
            for (auto& ns : idx.lookup_namespace(cur))
                result.push_back(ns);
        if (!std::is_same<T, cpp_namespace>::value)
            get_impl(std::false_type{}, result, idx);
    }

    void get_impl(std::false_type, std::vector<collie::ts::object_ref<const T>>& result,
                  const cpp_entity_index& idx) const
    {
        for (auto& cur : id())
        {
            auto entity = idx.lookup(cur).map([](const cpp_entity& e) {
                DEBUG_ASSERT(Predicate{}(e), detail::precondition_error_handler{},
                             "invalid entity type");
                return collie::ts::ref(static_cast<const T&>(e));
            });
            if (entity)
                result.push_back(collie::ts::ref(entity.value()));
        }
    }

    collie::ts::variant<cpp_entity_id, std::vector<cpp_entity_id>> target_;
    std::string                                                   name_;
};

/// \exclude
namespace detail
{
    struct cpp_entity_ref_predicate
    {
        bool operator()(const cpp_entity&)
        {
            return true;
        }
    };
} // namespace detail

/// A [hercules::ccast::basic_cpp_entity_ref]() to any [hercules::ccast::cpp_entity]().
using cpp_entity_ref = basic_cpp_entity_ref<cpp_entity, detail::cpp_entity_ref_predicate>;
} // namespace hercules::ccast

