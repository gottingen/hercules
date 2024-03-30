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

#include <memory>

#include <collie/type_safe/optional_ref.h>

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_entity_container.h>
#include <hercules/ast/cc/cpp_entity_index.h>
#include <hercules/ast/cc/cpp_expression.h>
#include <hercules/ast/cc/cpp_forward_declarable.h>
#include <hercules/ast/cc/cpp_type.h>

namespace hercules::ccast {
    /// A [hercules::ccast::cpp_entity]() modelling the value of an [hercules::ccast::cpp_enum]().
    class cpp_enum_value final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly created and registered enum value.
        /// \notes `value` may be `nullptr`, in which case the enum has an implicit value.
        static std::unique_ptr<cpp_enum_value> build(const cpp_entity_index &idx, cpp_entity_id id,
                                                     std::string name,
                                                     std::unique_ptr<cpp_expression> value = nullptr);

        /// \returns A [ts::optional_ref]() to the [hercules::ccast::cpp_expression]() that is the enum value.
        /// \notes It only has an associated expression if the value is explictly given.
        collie::ts::optional_ref<const cpp_expression> value() const noexcept {
            return collie::ts::opt_cref(value_.get());
        }

    private:
        cpp_enum_value(std::string name, std::unique_ptr<cpp_expression> value)
                : cpp_entity(std::move(name)), value_(std::move(value)) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        std::unique_ptr<cpp_expression> value_;
    };

    /// A [hercules::ccast::cpp_entity]() modelling a C++ enumeration.
    ///
    /// This can either be a definition or just a forward declaration.
    /// If it is just forward declared, it will not have any children.
    class cpp_enum final : public cpp_entity,
                           public cpp_entity_container<cpp_enum, cpp_enum_value>,
                           public cpp_forward_declarable {
    public:
        static cpp_entity_kind kind() noexcept;

        /// Builds a [hercules::ccast::cpp_enum]().
        class builder {
        public:
            /// \effects Sets the name, underlying type and whether it is scoped.
            builder(std::string name, bool scoped, std::unique_ptr<cpp_type> type, bool explicit_type)
                    : enum_(new cpp_enum(std::move(name), std::move(type), explicit_type, scoped)) {}

            /// \effects Adds a [hercules::ccast::cpp_enum_value]().
            void add_value(std::unique_ptr<cpp_enum_value> value) {
                enum_->add_child(std::move(value));
            }

            /// \returns The not yet finished enumeration.
            cpp_enum &get() noexcept {
                return *enum_;
            }

            /// \effects Registers the enum in the [hercules::ccast::cpp_entity_index](),
            /// using the given [hercules::ccast::cpp_entity_id]().
            /// \returns The finished enum.
            std::unique_ptr<cpp_enum> finish(
                    const cpp_entity_index &idx, cpp_entity_id id,
                    collie::ts::optional<cpp_entity_ref> semantic_parent) noexcept {
                enum_->set_semantic_parent(std::move(semantic_parent));
                idx.register_definition(std::move(id), collie::ts::ref(*enum_));
                return std::move(enum_);
            }

            /// \effects Marks the enum as forward declaration.
            /// \returns The finished enum.
            std::unique_ptr<cpp_enum> finish_declaration(const cpp_entity_index &idx,
                                                         cpp_entity_id definition_id) noexcept {
                enum_->mark_declaration(definition_id);
                idx.register_forward_declaration(std::move(definition_id), collie::ts::ref(*enum_));
                return std::move(enum_);
            }

        private:
            std::unique_ptr<cpp_enum> enum_;
        };

        /// \returns A reference to the underlying [hercules::ccast::cpp_type]() of the enum.
        const cpp_type &underlying_type() const noexcept {
            return *type_;
        }

        /// \returns Whether or not the underlying type is explictly given.
        bool has_explicit_type() const noexcept {
            return type_given_;
        }

        /// \returns Whether or not it is a scoped enumeration (i.e. an `enum class`).
        bool is_scoped() const noexcept {
            return scoped_;
        }

    private:
        cpp_enum(std::string name, std::unique_ptr<cpp_type> type, bool type_given, bool scoped)
                : cpp_entity(std::move(name)), type_(std::move(type)), scoped_(scoped), type_given_(type_given) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        collie::ts::optional<cpp_scope_name> do_get_scope_name() const override;

        std::unique_ptr<cpp_type> type_;
        bool scoped_, type_given_;
    };
} // namespace hercules::ccast
