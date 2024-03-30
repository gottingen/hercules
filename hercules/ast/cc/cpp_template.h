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

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_entity_container.h>
#include <hercules/ast/cc/cpp_template_parameter.h>
#include <hercules/ast/cc/cpp_token.h>

namespace hercules::ccast {
    /// Base class for all entities modelling a C++ template of some kind, aside from concepts
    ///
    /// It is a container of a single [hercules::ccast::cpp_entity]() that is the entity being templated.
    class cpp_template : public cpp_entity, public cpp_entity_container<cpp_template, cpp_entity> {
    public:
        /// \returns An iteratable object iterating over the [hercules::ccast::cpp_template_parameter]()
        /// entities. \notes These may be empty for a full specialization.
        detail::iteratable_intrusive_list<cpp_template_parameter> parameters() const noexcept {
            return collie::ts::ref(parameters_);
        }

    protected:
        /// Builder class for templates.
        ///
        /// Inherit from it to provide additional setter.
        template<class T, class EntityT>
        class basic_builder {
        public:
            /// \effects Sets the entity that is begin templated.
            basic_builder(std::unique_ptr<EntityT> templ) : template_entity(new T(std::move(templ))) {}

            basic_builder(basic_builder &&) = default;

            /// \effects Adds a parameter.
            void add_parameter(std::unique_ptr<cpp_template_parameter> parameter) {
                static_cast<cpp_template &>(*template_entity)
                        .parameters_.push_back(*template_entity, std::move(parameter));
            }

            /// \returns The not yet finished template.
            T &get() const noexcept {
                return *template_entity;
            }

            /// \effects Registers the template.
            /// \returns The finished template.
            std::unique_ptr<T> finish(const cpp_entity_index &idx, cpp_entity_id id, bool is_definition) {
                if (is_definition)
                    idx.register_definition(std::move(id), collie::ts::cref(*template_entity));
                else
                    idx.register_forward_declaration(std::move(id), collie::ts::cref(*template_entity));
                return std::move(template_entity);
            }

        protected:
            basic_builder() = default;

            ~basic_builder() noexcept = default;

            std::unique_ptr<T> template_entity;
        };

        /// \effects Sets the entity to be templated.
        cpp_template(std::unique_ptr<cpp_entity> entity) : cpp_entity(entity->name()) {
            add_child(std::move(entity));
        }

    private:
        collie::ts::optional<hercules::ccast::cpp_scope_name> do_get_scope_name() const override {
            return begin()->scope_name()
                   ? collie::ts::make_optional(hercules::ccast::cpp_scope_name(collie::ts::ref(*this)))
                   : collie::ts::nullopt;
        }

        detail::intrusive_list<cpp_template_parameter> parameters_;
    };

    /// A [hercules::ccast::cpp_type]() representing an instantiation of a [hercules::ccast::cpp_template]().
    class cpp_template_instantiation_type final : public cpp_type {
    public:
        /// Builds a [hercules::ccast::cpp_template_instantiation]().
        class builder {
        public:
            /// \effects Sets the primary template being instantiated.
            builder(cpp_template_ref templ)
                    : result_(new cpp_template_instantiation_type(std::move(templ))) {}

            /// \effects Adds the next argument.
            /// \requires No call to `add_unexposed_arguments()` has happened before.
            void add_argument(cpp_template_argument arg) {
                result_->arguments_.value(collie::ts::variant_type<std::vector<cpp_template_argument>>{})
                        .push_back(std::move(arg));
            }

            /// \effects Adds unexposed arguments as string.
            void add_unexposed_arguments(std::string arg) {
                result_->arguments_ = std::move(arg);
            }

            /// \returns The finished instantiation.
            std::unique_ptr<cpp_template_instantiation_type> finish() {
                return std::move(result_);
            }

        private:
            std::unique_ptr<cpp_template_instantiation_type> result_;
        };

        /// \returns A reference to the template that is being instantiated.
        /// \notes It could also point to a specialization,
        /// this is just the *primary* template.
        const cpp_template_ref &primary_template() const noexcept {
            return templ_;
        }

        /// \returns Whether or not the arguments are exposed.
        bool arguments_exposed() const noexcept {
            return arguments_.has_value(collie::ts::variant_type<std::vector<cpp_template_argument>>{});
        }

        /// \returns An array ref to the [hercules::ccast::cpp_template_argument](), if there are any.
        /// \requires The arguments are exposed, i.e. `arguments_exposed()` returns `true`.
        collie::ts::optional<collie::ts::array_ref<const cpp_template_argument>> arguments()
        const noexcept {
            auto &vec = arguments_.value(collie::ts::variant_type<std::vector<cpp_template_argument>>{});
            if (vec.empty())
                return collie::ts::nullopt;
            return collie::ts::ref(vec.data(), vec.size());
        }

        /// \returns The unexposed arguments as string.
        /// \requires The arguments are not exposed, i.e. `arguments_exposed()` returns `false`.
        const std::string &unexposed_arguments() const noexcept {
            return arguments_.value(collie::ts::variant_type<std::string>{});
        }

    private:
        cpp_template_instantiation_type(cpp_template_ref ref)
                : arguments_(collie::ts::variant_type<std::vector<cpp_template_argument>>{}),
                  templ_(std::move(ref)) {}

        cpp_type_kind do_get_kind() const noexcept override {
            return cpp_type_kind::template_instantiation_t;
        }

        collie::ts::variant<std::vector<cpp_template_argument>, std::string> arguments_;
        cpp_template_ref templ_;
    };

    /// Base class for all entities modelling a C++ template specialization.
    class cpp_template_specialization : public cpp_template {
    public:
        /// \returns A reference to the template that is being specialized.
        cpp_template_ref primary_template() const noexcept {
            return cpp_template_ref(templ_, name());
        }

        /// \returns Whether or not the arguments are exposed.
        bool arguments_exposed() const noexcept {
            return arguments_.has_value(collie::ts::variant_type<std::vector<cpp_template_argument>>{});
        }

        /// \returns An iteratable object iterating over the [hercules::ccast::cpp_template_argument]()s.
        /// \requires The arguments are exposed, i.e. `arguments_exposed()` returns `true`.
        /// \notes For function template specializations it can be empty,
        /// meaning that the arguments are not explictly given but deduced from the signature.
        collie::ts::array_ref<const cpp_template_argument> arguments() const noexcept {
            auto &vec = arguments_.value(collie::ts::variant_type<std::vector<cpp_template_argument>>{});
            return collie::ts::ref(vec.data(), vec.size());
        }

        /// \returns The unexposed arguments as string.
        /// \requires The arguments are not exposed, i.e. `arguments_exposed()` returns `false`.
        /// \notes For function template specializations it can be empty,
        /// meaning that the arguments are not explictly given but deduced from the signature.
        const cpp_token_string &unexposed_arguments() const noexcept {
            return arguments_.value(collie::ts::variant_type<cpp_token_string>{});
        }

        /// \returns Whether or not the specialization is a full specialization.
        bool is_full_specialization() const noexcept {
            // if no template parameters are given, it is a full specialization
            return parameters().empty();
        }

    protected:
        /// Builder class for specializations.
        ///
        /// Inherit from it to provide additional setter.
        template<class T, class EntityT>
        class specialization_builder : public basic_builder<T, EntityT> {
        public:
            /// \effects Sets the entity that is being templated and the primary template.
            specialization_builder(std::unique_ptr<EntityT> entity, const cpp_template_ref &templ) {
                this->template_entity = std::unique_ptr<T>(new T(std::move(entity), templ));
            }

            /// \effects Adds the next argument for the [hercules::ccast::cpp_template_parameter]() of the
            /// primary template. \requires No call to `add_unexposed_arguments()` has happened before.
            void add_argument(cpp_template_argument arg) {
                auto &specialization
                        = static_cast<cpp_template_specialization &>(*this->template_entity);
                specialization.arguments_
                        .value(collie::ts::variant_type<std::vector<cpp_template_argument>>{})
                        .push_back(std::move(arg));
            }

            /// \effects Adds unexposed arguments as string.
            void add_unexposed_arguments(cpp_token_string arg) {
                auto &specialization
                        = static_cast<cpp_template_specialization &>(*this->template_entity);
                specialization.arguments_ = std::move(arg);
            }

        protected:
            specialization_builder() = default;
        };

        /// \effects Sets the entity that is being templated and the primary template.
        cpp_template_specialization(std::unique_ptr<cpp_entity> entity, const cpp_template_ref &templ)
                : cpp_template(std::move(entity)),
                  arguments_(collie::ts::variant_type<std::vector<cpp_template_argument>>{}),
                  templ_(templ.id()[0u]) {
            DEBUG_ASSERT(!templ.is_overloaded()
                         && (templ.name().empty() || templ.name() == begin()->name()),
                         detail::precondition_error_handler{}, "invalid name of template ref");
        }

    private:
        collie::ts::variant<std::vector<cpp_template_argument>, cpp_token_string> arguments_;
        cpp_entity_id templ_;
    };
} // namespace hercules::ccast

