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

#include <collie/type_safe/optional.h>

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_file.h>

namespace hercules::ccast {
    /// A [hercules::ccast::cpp_entity]() modelling a macro parameter.
    class cpp_macro_parameter final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly built macro parameter.
        /// \notes It is not meant to be registered in the [hercules::ccast::cpp_entity_index]() as no other
        /// [hercules::ccast::cpp_entity]() can refer to it.
        static std::unique_ptr<cpp_macro_parameter> build(std::string name) {
            return std::unique_ptr<cpp_macro_parameter>(new cpp_macro_parameter(std::move(name)));
        }

    private:
        cpp_macro_parameter(std::string name) : cpp_entity(std::move(name)) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;
    };

    /// A [hercules::ccast::cpp_entity]() modelling a macro definition.
    class cpp_macro_definition final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly built object like macro.
        /// \notes It is not meant to be registered in the [hercules::ccast::cpp_entity_index](),
        /// as no other [hercules::ccast::cpp_entity]() can refer to it.
        static std::unique_ptr<cpp_macro_definition> build_object_like(std::string name,
                                                                       std::string replacement) {
            std::unique_ptr<cpp_macro_definition> result{new cpp_macro_definition(std::move(name))};
            result->replacement_ = std::move(replacement);
            return result;
        }

        /// Builds a function like macro.
        class function_like_builder {
        public:
            /// \effects Sets the name of the function like macro.
            function_like_builder(std::string name) : result_(new cpp_macro_definition(std::move(name))) {
                result_->kind_ = function_like;
            }

            /// \effects Sets the replacement text.
            void replacement(std::string replacement) {
                result_->replacement_ = std::move(replacement);
            }

            /// \effects Marks the macro as variadic.
            void is_variadic() {
                result_->kind_ = variadic_function;
            }

            /// \effects Adds a parameter.
            /// \group param
            void parameter(std::unique_ptr<cpp_macro_parameter> param) {
                result_->parameters_.push_back(*result_, std::move(param));
            }

            /// \group param
            void parameter(std::string name) {
                parameter(cpp_macro_parameter::build(std::move(name)));
            }

            /// \returns The finished macro.
            /// \notes It is not meant to be registered in the [hercules::ccast::cpp_entity_index](),
            /// as no other [hercules::ccast::cpp_entity]() can refer to it.
            std::unique_ptr<cpp_macro_definition> finish() {
                return std::move(result_);
            }

        private:
            std::unique_ptr<cpp_macro_definition> result_;
        };

        /// \returns The replacement text of the macro.
        const std::string &replacement() const noexcept {
            return replacement_;
        }

        /// \returns Whether or not it is an object like macro.
        bool is_object_like() const noexcept {
            return kind_ == object_like;
        }

        /// \returns Whether or not it is a function like macro.
        bool is_function_like() const noexcept {
            return kind_ != object_like;
        }

        /// \returns Whether or not it is a variadic macro.
        bool is_variadic() const noexcept {
            return kind_ == variadic_function;
        }

        /// \returns The parameters of the macro.
        /// \notes It has none if it is not a function like macro.
        detail::iteratable_intrusive_list<cpp_macro_parameter> parameters() const noexcept {
            return collie::ts::ref(parameters_);
        }

    private:
        cpp_entity_kind do_get_entity_kind() const noexcept override;

        cpp_macro_definition(std::string name) : cpp_entity(std::move(name)), kind_(object_like) {}

        detail::intrusive_list<cpp_macro_parameter> parameters_;
        std::string replacement_;

        enum : char {
            object_like,
            function_like,
            variadic_function,
        } kind_;

        friend function_like_builder;
    };

    /// The kind of [hercules::ccast::cpp_include_directive]().
    enum class cpp_include_kind {
        system, //< An `#include <...>`.
        local,  //< An `#include "..."`.
    };

    /// A [hercules::ccast::cpp_entity]() modelling an `#include`.
    class cpp_include_directive final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly built include directive.
        /// \notes It is not meant to be registered in the [hercules::ccast::cpp_entity_index](),
        /// as no other [hercules::ccast::cpp_entity]() can refer to it.
        static std::unique_ptr<cpp_include_directive> build(const cpp_file_ref &target,
                                                            cpp_include_kind kind,
                                                            std::string full_path) {
            return std::unique_ptr<cpp_include_directive>(
                    new cpp_include_directive(target, kind, std::move(full_path)));
        }

        /// \returns A reference to the [hercules::ccast::cpp_file]() it includes.
        cpp_file_ref target() const noexcept {
            return cpp_file_ref(target_, name());
        }

        /// \returns The kind of include it is.
        cpp_include_kind include_kind() const noexcept {
            return kind_;
        }

        /// \returns The full path of the included file.
        const std::string &full_path() const noexcept {
            return full_path_;
        }

    private:
        cpp_entity_kind do_get_entity_kind() const noexcept override;

        cpp_include_directive(const cpp_file_ref &target, cpp_include_kind kind, std::string full_path)
                : cpp_entity(target.name()), target_(target.id()[0u]), kind_(kind),
                  full_path_(std::move(full_path)) {
            DEBUG_ASSERT(!target.is_overloaded(), detail::precondition_error_handler{});
        }

        cpp_entity_id target_;
        cpp_include_kind kind_;
        std::string full_path_;
    };
} // namespace hercules::ccast
