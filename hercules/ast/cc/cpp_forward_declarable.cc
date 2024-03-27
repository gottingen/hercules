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


#include <hercules/ast/cc/cpp_forward_declarable.h>

#include <hercules/ast/cc/cpp_class.h>
#include <hercules/ast/cc/cpp_entity_kind.h>
#include <hercules/ast/cc/cpp_enum.h>
#include <hercules/ast/cc/cpp_function.h>
#include <hercules/ast/cc/cpp_template.h>
#include <hercules/ast/cc/cpp_variable.h>

using namespace hercules::ccast;

namespace
{
collie::ts::optional_ref<const cpp_forward_declarable> get_declarable(const cpp_entity& e)
{
    switch (e.kind())
    {
    case cpp_entity_kind::enum_t:
        return collie::ts::ref(static_cast<const cpp_enum&>(e));
    case cpp_entity_kind::class_t:
        return collie::ts::ref(static_cast<const cpp_class&>(e));
    case cpp_entity_kind::variable_t:
        return collie::ts::ref(static_cast<const cpp_variable&>(e));
    case cpp_entity_kind::function_t:
    case cpp_entity_kind::member_function_t:
    case cpp_entity_kind::conversion_op_t:
    case cpp_entity_kind::constructor_t:
    case cpp_entity_kind::destructor_t:
        return collie::ts::ref(static_cast<const cpp_function_base&>(e));
    case cpp_entity_kind::function_template_t:
    case cpp_entity_kind::function_template_specialization_t:
    case cpp_entity_kind::class_template_t:
    case cpp_entity_kind::class_template_specialization_t:
        return get_declarable(*static_cast<const cpp_template&>(e).begin());

    case cpp_entity_kind::file_t:
    case cpp_entity_kind::macro_parameter_t:
    case cpp_entity_kind::macro_definition_t:
    case cpp_entity_kind::include_directive_t:
    case cpp_entity_kind::language_linkage_t:
    case cpp_entity_kind::namespace_t:
    case cpp_entity_kind::namespace_alias_t:
    case cpp_entity_kind::using_directive_t:
    case cpp_entity_kind::using_declaration_t:
    case cpp_entity_kind::type_alias_t:
    case cpp_entity_kind::enum_value_t:
    case cpp_entity_kind::access_specifier_t:
    case cpp_entity_kind::base_class_t:
    case cpp_entity_kind::member_variable_t:
    case cpp_entity_kind::bitfield_t:
    case cpp_entity_kind::function_parameter_t:
    case cpp_entity_kind::friend_t:
    case cpp_entity_kind::template_type_parameter_t:
    case cpp_entity_kind::non_type_template_parameter_t:
    case cpp_entity_kind::template_template_parameter_t:
    case cpp_entity_kind::concept_t:
    case cpp_entity_kind::alias_template_t:
    case cpp_entity_kind::variable_template_t:
    case cpp_entity_kind::static_assert_t:
    case cpp_entity_kind::unexposed_t:
        return nullptr;

    case cpp_entity_kind::count:
        break;
    }

    DEBUG_UNREACHABLE(detail::assert_handler{});
    return nullptr;
}

collie::ts::optional_ref<const cpp_entity> get_definition_impl(const cpp_entity_index& idx,
                                                              const cpp_entity&       e)
{
    auto declarable = get_declarable(e);
    if (!declarable || declarable.value().is_definition())
        // not declarable or is a definition
        // return reference to entity itself
        return collie::ts::ref(e);
    // else lookup definition
    return idx.lookup_definition(declarable.value().definition().value());
}
} // namespace

bool hercules::ccast::is_definition(const cpp_entity& e) noexcept
{
    auto declarable = get_declarable(e);
    return declarable && declarable.value().is_definition();
}

collie::ts::optional_ref<const cpp_entity> hercules::ccast::get_definition(const cpp_entity_index& idx,
                                                                 const cpp_entity&       e)
{
    return get_definition_impl(idx, e);
}

collie::ts::optional_ref<const cpp_enum> hercules::ccast::get_definition(const cpp_entity_index& idx,
                                                               const cpp_enum&         e)
{
    return get_definition_impl(idx, e).map([](const cpp_entity& e) {
        DEBUG_ASSERT(e.kind() == cpp_entity_kind::enum_t, detail::assert_handler{});
        return collie::ts::ref(static_cast<const cpp_enum&>(e));
    });
}

collie::ts::optional_ref<const cpp_class> hercules::ccast::get_definition(const cpp_entity_index& idx,
                                                                const cpp_class&        e)
{
    return get_definition_impl(idx, e).map([](const cpp_entity& e) {
        DEBUG_ASSERT(e.kind() == cpp_entity_kind::class_t, detail::assert_handler{});
        return collie::ts::ref(static_cast<const cpp_class&>(e));
    });
}

collie::ts::optional_ref<const cpp_variable> hercules::ccast::get_definition(const cpp_entity_index& idx,
                                                                   const cpp_variable&     e)
{
    return get_definition_impl(idx, e).map([](const cpp_entity& e) {
        DEBUG_ASSERT(e.kind() == cpp_entity_kind::variable_t, detail::assert_handler{});
        return collie::ts::ref(static_cast<const cpp_variable&>(e));
    });
}

collie::ts::optional_ref<const cpp_function_base> hercules::ccast::get_definition(const cpp_entity_index& idx,
                                                                        const cpp_function_base& e)
{
    return get_definition_impl(idx, e).map([](const cpp_entity& e) {
        DEBUG_ASSERT(is_function(e.kind()), detail::assert_handler{});
        return collie::ts::ref(static_cast<const cpp_function_base&>(e));
    });
}
