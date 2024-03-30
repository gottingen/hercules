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

#include <hercules/ast/cc/cpp_entity_container.h>
#include <hercules/ast/cc/cpp_entity_index.h>
#include <hercules/ast/cc/cpp_entity_kind.h>
#include <hercules/ast/cc/cpp_entity_ref.h>

namespace hercules::ccast {
    /// A [hercules::ccast::cpp_entity]() modelling a namespace.
    class cpp_namespace final : public cpp_entity,
                                public cpp_entity_container<cpp_namespace, cpp_entity> {
    public:
        static cpp_entity_kind kind() noexcept;

        /// Builds a [hercules::ccast::cpp_namespace]().
        class builder {
        public:
            /// \effects Sets the namespace name and whether it is inline and nested.
            explicit builder(std::string name, bool is_inline, bool is_nested)
                    : namespace_(new cpp_namespace(std::move(name), is_inline, is_nested)) {}

            /// \effects Adds an entity.
            void add_child(std::unique_ptr<cpp_entity> child) noexcept {
                namespace_->add_child(std::move(child));
            }

            /// \returns The not yet finished namespace.
            cpp_namespace &get() const noexcept {
                return *namespace_;
            }

            /// \effects Registers the namespace in the [hercules::ccast::cpp_entity_index](),
            /// using the given [hercules::ccast::cpp_entity_id]().
            /// \returns The finished namespace.
            std::unique_ptr<cpp_namespace> finish(const cpp_entity_index &idx, cpp_entity_id id) {
                idx.register_namespace(std::move(id), collie::ts::ref(*namespace_));
                return std::move(namespace_);
            }

        private:
            std::unique_ptr<cpp_namespace> namespace_;
        };

        /// \returns Whether or not the namespace is an `inline namespace`.
        bool is_inline() const noexcept {
            return inline_;
        }

        /// \returns Whether or not the namespace is part of a C++17 nested namespace.
        bool is_nested() const noexcept {
            return nested_;
        }

        /// \returns Whether or not the namespace is anonymous.
        bool is_anonymous() const noexcept {
            return name().empty();
        }

    private:
        cpp_namespace(std::string name, bool is_inline, bool is_nested)
                : cpp_entity(std::move(name)), inline_(is_inline), nested_(is_nested) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        collie::ts::optional<cpp_scope_name> do_get_scope_name() const override {
            return collie::ts::ref(*this);
        }

        bool inline_;
        bool nested_;
    };

/// \exclude
    namespace detail {
        struct cpp_namespace_ref_predicate {
            bool operator()(const cpp_entity &e);
        };
    } // namespace detail

/// A reference to a [hercules::ccast::cpp_namespace]().
    using cpp_namespace_ref = basic_cpp_entity_ref<cpp_namespace, detail::cpp_namespace_ref_predicate>;

/// A [hercules::ccast::cpp_entity]() modelling a namespace alias.
    class cpp_namespace_alias final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly created and registered namespace alias.
        static std::unique_ptr<cpp_namespace_alias> build(const cpp_entity_index &idx, cpp_entity_id id,
                                                          std::string name, cpp_namespace_ref target);

        /// \returns The [hercules::ccast::cpp_namespace_ref]() to the aliased namespace.
        /// \notes If the namespace aliases aliases another namespace alias,
        /// the target entity will still be the namespace, not the alias.
        const cpp_namespace_ref &target() const noexcept {
            return target_;
        }

    private:
        cpp_namespace_alias(std::string name, cpp_namespace_ref target)
                : cpp_entity(std::move(name)), target_(std::move(target)) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        cpp_namespace_ref target_;
    };

/// A [hercules::ccast::cpp_entity]() modelling a using directive.
///
/// A using directive is `using namespace std`, for example.
/// \notes It does not have a name.
    class cpp_using_directive final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly created using directive.
        /// \notes It is not meant to be registered at the [hercules::ccast::cpp_entity_index](),
        /// as nothing can refer to it.
        static std::unique_ptr<cpp_using_directive> build(cpp_namespace_ref target) {
            return std::unique_ptr<cpp_using_directive>(new cpp_using_directive(std::move(target)));
        }

        /// \returns The [hercules::ccast::cpp_namespace_ref]() that is being used.
        const cpp_namespace_ref &target() const {
            return target_;
        }

    private:
        cpp_using_directive(cpp_namespace_ref target) : cpp_entity(""), target_(std::move(target)) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        cpp_namespace_ref target_;
    };

/// A [hercules::ccast::cpp_entity]() modelling a using declaration.
///
/// A using declaration is `using std::vector`, for example.
/// \notes It does not have a name.
    class cpp_using_declaration final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly created using declaration.
        /// \notes It is not meant to be registered at the [hercules::ccast::cpp_entity_index](),
        /// as nothing can refer to it.
        static std::unique_ptr<cpp_using_declaration> build(cpp_entity_ref target) {
            return std::unique_ptr<cpp_using_declaration>(new cpp_using_declaration(std::move(target)));
        }

        /// \returns The [hercules::ccast::cpp_entity_ref]() that is being used.
        /// \notes The name of the reference is the same as the name of this entity.
        const cpp_entity_ref &target() const noexcept {
            return target_;
        }

    private:
        cpp_using_declaration(cpp_entity_ref target) : cpp_entity(""), target_(std::move(target)) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        cpp_entity_ref target_;
    };
} // namespace hercules::ccast
