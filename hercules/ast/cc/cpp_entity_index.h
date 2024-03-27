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

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <collie/type_safe/optional_ref.h>
#include <collie/type_safe/reference.h>
#include <collie/type_safe/strong_typedef.h>

#include <hercules/ast/cc/cppast_fwd.h>

namespace hercules::ccast
{
/// \exclude
namespace detail
{
    using hash_type               = std::uint_least64_t;
    constexpr hash_type fnv_basis = 14695981039346656037ull;
    constexpr hash_type fnv_prime = 1099511628211ull;

    // FNV-1a 64 bit hash
    constexpr hash_type id_hash(const char* str, hash_type hash = fnv_basis)
    {
        return *str ? id_hash(str + 1, (hash ^ hash_type(*str)) * fnv_prime) : hash;
    }
} // namespace detail

/// A [ts::strong_typedef]() representing the unique id of a [hercules::ccast::cpp_entity]().
///
/// It is comparable for equality.
struct cpp_entity_id : collie::ts::strong_typedef<cpp_entity_id, detail::hash_type>,
                       collie::ts::strong_typedef_op::equality_comparison<cpp_entity_id>
{
    explicit cpp_entity_id(const std::string& str) : cpp_entity_id(str.c_str()) {}

    explicit cpp_entity_id(const char* str) : strong_typedef(detail::id_hash(str)) {}
};

inline namespace literals
{
    /// \returns A new [hercules::ccast::cpp_entity_id]() created from the given string.
    inline cpp_entity_id operator"" _id(const char* str, std::size_t)
    {
        return cpp_entity_id(str);
    }
} // namespace literals

/// An index of all [hercules::ccast::cpp_entity]() objects created.
///
/// It maps [hercules::ccast::cpp_entity_id]() to references to the [hercules::ccast::cpp_entity]() objects.
class cpp_entity_index
{
public:
    /// Exception thrown on duplicate entity definition.
    class duplicate_definition_error : public std::logic_error
    {
    public:
        duplicate_definition_error();
    };

    /// \effects Registers a new [hercules::ccast::cpp_entity]() which is a definition.
    /// It will override any previously registered declarations of the same entity.
    /// \throws duplicate_defintion_error if the entity has been registered as definition before.
    /// \requires The entity must live as long as the index lives,
    /// and it must not be a namespace.
    /// \notes This operation is thread safe.
    void register_definition(cpp_entity_id                           id,
                             collie::ts::object_ref<const cpp_entity> entity) const;

    /// \effects Registers a new [hercules::ccast::cpp_file]().
    /// \returns `true` if the file was not registered before.
    /// If it returns `false`, the file was registered before and nothing was changed.
    /// \requires The entity must live as long as the index lives.
    /// \notes This operation is thread safe.
    bool register_file(cpp_entity_id id, collie::ts::object_ref<const cpp_file> file) const;

    /// \effects Registers a new [hercules::ccast::cpp_entity]() which is a declaration.
    /// Only the first declaration will be registered.
    /// \requires The entity must live as long as the index lives.
    /// \requires The entity must be forward declarable.
    /// \notes This operation is thread safe.
    void register_forward_declaration(cpp_entity_id                           id,
                                      collie::ts::object_ref<const cpp_entity> entity) const;

    /// \effects Registers a new [hercules::ccast::cpp_namespace]().
    /// \notes The namespace object must live as long as the index lives.
    /// \notes This operation is thread safe.
    void register_namespace(cpp_entity_id id, collie::ts::object_ref<const cpp_namespace> ns) const;

    /// \returns A [ts::optional_ref]() corresponding to the entity(/ies) of the given
    /// [hercules::ccast::cpp_entity_id](). If no definition has been registered, it return the first
    /// declaration that was registered. If the id resolves to a namespaces, returns an empty
    /// optional. \notes This operation is thread safe.
    collie::ts::optional_ref<const cpp_entity> lookup(const cpp_entity_id& id) const noexcept;

    /// \returns A [ts::optional_ref]() corresponding to the entity of the given
    /// [hercules::ccast::cpp_entity_id](). If no definition has been registered, it returns an empty
    /// optional. \notes This operation is thread safe.
    collie::ts::optional_ref<const cpp_entity> lookup_definition(
        const cpp_entity_id& id) const noexcept;

    /// \returns A [ts::array_ref]() of references to all namespaces matching the given
    /// [hercules::ccast::cpp_entity_id](). If no namespace is found, it returns an empty array reference.
    /// \notes This operation is thread safe.
    auto lookup_namespace(const cpp_entity_id& id) const noexcept
        -> collie::ts::array_ref<collie::ts::object_ref<const cpp_namespace>>;

private:
    struct hash
    {
        std::size_t operator()(const cpp_entity_id& id) const noexcept
        {
            return std::size_t(static_cast<detail::hash_type>(id));
        }
    };

    struct value
    {
        collie::ts::object_ref<const cpp_entity> entity;
        bool                                    is_definition;

        value(collie::ts::object_ref<const cpp_entity> e, bool def)
        : entity(std::move(e)), is_definition(def)
        {}
    };

    mutable std::mutex                                     mutex_;
    mutable std::unordered_map<cpp_entity_id, value, hash> map_;
    mutable std::unordered_map<cpp_entity_id,
                               std::vector<collie::ts::object_ref<const cpp_namespace>>, hash>
        ns_;
};
} // namespace hercules::ccast

