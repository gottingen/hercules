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

#include <hercules/ast/cc/cppast_fwd.h>
#include <hercules/ast/cc/detail/assert.h>

namespace hercules::ccast {
    /// All possible kinds of C++ entities.
    enum class cpp_entity_kind {
        file_t,

        macro_parameter_t,
        macro_definition_t,
        include_directive_t,

        language_linkage_t,

        namespace_t,
        namespace_alias_t,
        using_directive_t,
        using_declaration_t,

        type_alias_t,

        enum_t,
        enum_value_t,

        class_t,
        access_specifier_t,
        base_class_t,

        variable_t,
        member_variable_t,
        bitfield_t,

        function_parameter_t,
        function_t,
        member_function_t,
        conversion_op_t,
        constructor_t,
        destructor_t,

        friend_t,

        template_type_parameter_t,
        non_type_template_parameter_t,
        template_template_parameter_t,

        alias_template_t,
        variable_template_t,
        function_template_t,
        function_template_specialization_t,
        class_template_t,
        class_template_specialization_t,
        concept_t,

        static_assert_t,

        unexposed_t,

        count,
    };

    /// \returns A human readable string describing the entity kind.
    const char *to_string(cpp_entity_kind kind) noexcept;

    /// \returns Whether or not a given entity kind is a C++ function,
    /// that is, it dervies from [hercules::ccast::cpp_function_base]().
    bool is_function(cpp_entity_kind kind) noexcept;

    /// \returns Whether or not a given entity kind is a C++ (template) parameter.
    bool is_parameter(cpp_entity_kind kind) noexcept;

    /// \returns Whether or not a given entity kind is a C++ template,
    /// that is, it dervies from [hercules::ccast::cpp_template]().
    /// \notes A template template parameter is not considered a template for this function.
    /// \notes Template specializations are also considered templates here.
    bool is_template(cpp_entity_kind kind) noexcept;

    /// \returns Whether or not a given entity kind is a specialization of a C++ template,
    /// that is, it derives from [hercules::ccast::cpp_template_specialization]().
    bool is_template_specialization(cpp_entity_kind kind) noexcept;
} // namespace hercules::ccast

