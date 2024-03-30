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

#include <type_traits>

#include <collie/type_safe/optional.h>

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_entity_ref.h>

namespace hercules::ccast {
    /// Mixin base class for all entities that can have a forward declaration.
    ///
    /// Examples are [hercules::ccast::cpp_enum]() or [hercules::ccast::cpp_class](),
    /// but also [hercules::ccast::cpp_function_base]().
    /// Those entities can have multiple declarations and one definition.
    class cpp_forward_declarable {
    public:
        /// \returns Whether or not the entity is the definition.
        bool is_definition() const noexcept {
            return !definition_.has_value();
        }

        /// \returns Whether or not the entity is "just" a declaration.
        bool is_declaration() const noexcept {
            return definition_.has_value();
        }

        /// \returns The [hercules::ccast::cpp_entity_id]() of the definition,
        /// if the current entity is not the definition.
        const collie::ts::optional<cpp_entity_id> &definition() const noexcept {
            return definition_;
        }

        /// \returns A reference to the semantic parent of the entity.
        /// This applies only to out-of-line definitions
        /// and is the entity which owns the declaration.
        const collie::ts::optional<cpp_entity_ref> &semantic_parent() const noexcept {
            return semantic_parent_;
        }

        /// \returns The name of the semantic parent, if it has one,
        /// else the empty string.
        /// \notes This may include template parameters.
        std::string semantic_scope() const noexcept {
            return collie::ts::copy(semantic_parent_.map(&cpp_entity_ref::name)).value_or("");
        }

    protected:
        /// \effects Marks the entity as definition.
        /// \notes If it is not a definition,
        /// [*set_definition]() must be called.
        cpp_forward_declarable() noexcept = default;

        ~cpp_forward_declarable() noexcept = default;

        /// \effects Sets the definition entity,
        /// marking it as a forward declaration.
        void mark_declaration(cpp_entity_id def) noexcept {
            definition_ = std::move(def);
        }

        /// \effects Sets the semantic parent of the entity.
        void set_semantic_parent(collie::ts::optional<cpp_entity_ref> semantic_parent) noexcept {
            semantic_parent_ = std::move(semantic_parent);
        }

    private:
        collie::ts::optional<cpp_entity_ref> semantic_parent_;
        collie::ts::optional<cpp_entity_id> definition_;
    };

    /// \returns Whether or not the given entity is a definition.
    bool is_definition(const cpp_entity &e) noexcept;

    /// Gets the definition of an entity.
    /// \returns A [ts::optional_ref]() to the entity that is the definition.
    /// If the entity is a definition or not derived from [hercules::ccast::cpp_forward_declarable]() (only valid
    /// for the generic entity overload), returns a reference to the entity itself. Otherwise lookups
    /// the definition id and returns it. \notes The return value will only be `nullptr`, if the
    /// definition is not registered. \group get_definition
    collie::ts::optional_ref<const cpp_entity> get_definition(const cpp_entity_index &idx,
                                                              const cpp_entity &e);

    /// \group get_definition
    collie::ts::optional_ref<const cpp_enum> get_definition(const cpp_entity_index &idx,
                                                            const cpp_enum &e);

    /// \group get_definition
    collie::ts::optional_ref<const cpp_class> get_definition(const cpp_entity_index &idx,
                                                             const cpp_class &e);

    /// \group get_definition
    collie::ts::optional_ref<const cpp_variable> get_definition(const cpp_entity_index &idx,
                                                                const cpp_variable &e);

    /// \group get_definition
    collie::ts::optional_ref<const cpp_function_base> get_definition(const cpp_entity_index &idx,
                                                                     const cpp_function_base &e);

} // namespace hercules::ccast
