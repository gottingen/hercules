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
#include <hercules/ast/cc/cpp_forward_declarable.h>
#include <hercules/ast/cc/cpp_storage_class_specifiers.h>
#include <hercules/ast/cc/cpp_variable_base.h>

namespace hercules::ccast {
    /// A [hercules::ccast::cpp_entity]() modelling a C++ variable.
    /// \notes This is not a member variable,
    /// use [hercules::ccast::cpp_member_variable]() for that.
    /// But it can be `static` member variable.
    class cpp_variable final : public cpp_entity,
                               public cpp_variable_base,
                               public cpp_forward_declarable {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly created and registered variable.
        /// \notes The default value may be `nullptr` indicating no default value.
        static std::unique_ptr<cpp_variable> build(const cpp_entity_index &idx, cpp_entity_id id,
                                                   std::string name, std::unique_ptr<cpp_type> type,
                                                   std::unique_ptr<cpp_expression> def,
                                                   cpp_storage_class_specifiers spec, bool is_constexpr,
                                                   collie::ts::optional<cpp_entity_ref> semantic_parent
                                                   = {});

        /// \returns A newly created variable that is a declaration.
        /// A declaration will not be registered and it does not have the default value.
        static std::unique_ptr<cpp_variable> build_declaration(
                cpp_entity_id definition_id, std::string name, std::unique_ptr<cpp_type> type,
                cpp_storage_class_specifiers spec, bool is_constexpr,
                collie::ts::optional<cpp_entity_ref> semantic_parent = {});

        /// \returns The [hercules::ccast::cpp_storage_specifiers]() on that variable.
        cpp_storage_class_specifiers storage_class() const noexcept {
            return storage_;
        }

        /// \returns Whether the variable is marked `constexpr`.
        bool is_constexpr() const noexcept {
            return is_constexpr_;
        }

    private:
        cpp_variable(std::string name, std::unique_ptr<cpp_type> type,
                     std::unique_ptr<cpp_expression> def, cpp_storage_class_specifiers spec,
                     bool is_constexpr)
                : cpp_entity(std::move(name)), cpp_variable_base(std::move(type), std::move(def)),
                  storage_(spec), is_constexpr_(is_constexpr) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        cpp_storage_class_specifiers storage_;
        bool is_constexpr_;
    };
} // namespace hercules::ccast
