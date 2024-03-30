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
#include <hercules/ast/cc/cpp_entity_container.h>
#include <hercules/ast/cc/cpp_forward_declarable.h>
#include <hercules/ast/cc/cpp_type.h>

namespace hercules::ccast {
    /// The keyword used on the declaration of a [hercules::ccast::cpp_class]().
    enum class cpp_class_kind {
        class_t,
        struct_t,
        union_t,
    };

    /// \returns The keyword as a string.
    const char *to_string(cpp_class_kind kind) noexcept;

    /// The C++ access specifiers.
    enum cpp_access_specifier_kind : int {
        cpp_public,
        cpp_protected,
        cpp_private
    };

    /// \returns The access specifier keyword as a string.
    const char *to_string(cpp_access_specifier_kind access) noexcept;

    /// A [hercules::ccast::cpp_entity]() modelling a C++ access specifier.
    class cpp_access_specifier final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly created access specifier.
        /// \notes It is not meant to be registered at the [hercules::ccast::cpp_entity_index](),
        /// as nothing can refer to it.
        static std::unique_ptr<cpp_access_specifier> build(cpp_access_specifier_kind kind) {
            return std::unique_ptr<cpp_access_specifier>(new cpp_access_specifier(kind));
        }

        /// \returns The kind of access specifier.
        cpp_access_specifier_kind access_specifier() const noexcept {
            return access_;
        }

    private:
        cpp_access_specifier(cpp_access_specifier_kind access)
                : cpp_entity(to_string(access)), access_(access) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        cpp_access_specifier_kind access_;
    };

    /// A [hercules::ccast::cpp_entity]() modelling a base class specifier.
    class cpp_base_class final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly created base class specifier.
        /// \notes It is not meant to be registered at the [hercules::ccast::cpp_entity_index](),
        /// as nothing can refer to the specifier itself.
        static std::unique_ptr<cpp_base_class> build(std::string name, std::unique_ptr<cpp_type> base,
                                                     cpp_access_specifier_kind access, bool is_virtual) {
            return std::unique_ptr<cpp_base_class>(
                    new cpp_base_class(std::move(name), std::move(base), access, is_virtual));
        }

        /// \returns The type of the base class.
        const cpp_type &type() const {
            return *type_;
        }

        /// \returns The access specifier of the base class.
        cpp_access_specifier_kind access_specifier() const noexcept {
            return access_;
        }

        /// \returns Whether or not it is a `virtual` base class.
        bool is_virtual() const noexcept {
            return virtual_;
        }

    private:
        cpp_base_class(std::string name, std::unique_ptr<cpp_type> base,
                       cpp_access_specifier_kind access, bool is_virtual)
                : cpp_entity(std::move(name)), type_(std::move(base)), access_(access), virtual_(is_virtual) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        std::unique_ptr<cpp_type> type_;
        cpp_access_specifier_kind access_;
        bool virtual_;
    };

/// A [hercules::ccast::cpp_entity]() modelling a C++ class.
///
/// This can either be a definition or just a forward declaration.
/// If it is just a forward declaration,
/// everything except the class type will not be available.
    class cpp_class final : public cpp_entity,
                            public cpp_entity_container<cpp_class, cpp_entity>,
                            public cpp_forward_declarable {
    public:
        static cpp_entity_kind kind() noexcept;

        /// Builds a [hercules::ccast::cpp_class]().
        class builder {
        public:
            /// \effects Sets the name and kind and whether it is `final`.
            explicit builder(std::string name, cpp_class_kind kind, bool is_final = false)
                    : class_(new cpp_class(std::move(name), kind, is_final)) {}

            /// \effects Marks the class as final.
            void is_final() noexcept {
                class_->final_ = true;
            }

            /// \effects Builds a [hercules::ccast::cpp_base_class]() and adds it.
            cpp_base_class &base_class(std::string name, std::unique_ptr<cpp_type> type,
                                       cpp_access_specifier_kind access, bool is_virtual) {
                return add_base_class(
                        cpp_base_class::build(std::move(name), std::move(type), access, is_virtual));
            }

            /// \effects Adds a new base class.
            cpp_base_class &add_base_class(std::unique_ptr<cpp_base_class> base) noexcept {
                auto bptr = base.get();
                class_->bases_.push_back(*class_, std::move(base));
                return *bptr;
            }

            /// \effects Builds a [hercules::ccast::cpp_access_specifier]() and adds it.
            void access_specifier(cpp_access_specifier_kind access) {
                add_child(cpp_access_specifier::build(access));
            }

            /// \effects Adds an entity.
            void add_child(std::unique_ptr<cpp_entity> child) noexcept {
                class_->add_child(std::move(child));
            }

            /// \returns The not yet finished class.
            cpp_class &get() noexcept {
                return *class_;
            }

            /// \effects Registers the class in the [hercules::ccast::cpp_entity_index](),
            /// using the given [hercules::ccast::cpp_entity_id]().
            /// \returns The finished class.
            std::unique_ptr<cpp_class> finish(const cpp_entity_index &idx, cpp_entity_id id,
                                              collie::ts::optional<cpp_entity_ref> semantic_parent);

            /// \effects Marks the class as forward declaration.
            /// \returns The finished class.
            std::unique_ptr<cpp_class> finish_declaration(const cpp_entity_index &idx,
                                                          cpp_entity_id definition_id);

            /// \effects Returns the finished class without registering it.
            /// \notes This is intended for templated classes only.
            std::unique_ptr<cpp_class> finish(collie::ts::optional<cpp_entity_ref> semantic_parent);

            /// \effects Returns the finish class without registering it and marks it as forward
            /// declaration. \notes This is intended for templated classes only.
            std::unique_ptr<cpp_class> finish_declaration(cpp_entity_id definition_id);

        private:
            std::unique_ptr<cpp_class> class_;
        };

        /// \returns The keyword used in the declaration of the class.
        cpp_class_kind class_kind() const noexcept {
            return kind_;
        }

        /// \returns Whether or not the class was declared `final`.
        bool is_final() const noexcept {
            return final_;
        }

        /// \returns An iteratable object iterating over the [hercules::ccast::cpp_base_class]() specifiers.
        detail::iteratable_intrusive_list<cpp_base_class> bases() const noexcept {
            return collie::ts::ref(bases_);
        }

    private:
        cpp_class(std::string name, cpp_class_kind kind, bool final)
                : cpp_entity(std::move(name)), kind_(kind), final_(final) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        collie::ts::optional<cpp_scope_name> do_get_scope_name() const override {
            return collie::ts::ref(*this);
        }

        detail::intrusive_list<cpp_base_class> bases_;
        cpp_class_kind kind_;
        bool final_;
    };

    /// \returns The type the base class refers to.
    /// It is either a class or some form of typedef.
    collie::ts::optional_ref<const cpp_entity> get_class_or_typedef(const cpp_entity_index &index,
                                                                    const cpp_base_class &base);

    /// \returns The type the base class refers to.
    /// Typedefs are unwrapped.
    collie::ts::optional_ref<const cpp_class> get_class(const cpp_entity_index &index,
                                                        const cpp_base_class &base);
} // namespace hercules::ccast
