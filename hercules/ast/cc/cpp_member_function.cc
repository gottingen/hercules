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


#include <hercules/ast/cc/cpp_member_function.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

using namespace hercules::ccast;

std::string cpp_member_function_base::do_get_signature() const
{
    auto result = cpp_function_base::do_get_signature();

    if (is_const(cv_qualifier()))
        result += " const";
    if (is_volatile(cv_qualifier()))
        result += " volatile";

    if (ref_qualifier() == cpp_ref_lvalue)
        result += " &";
    else if (ref_qualifier() == cpp_ref_rvalue)
        result += " &&";

    return result;
}

cpp_entity_kind cpp_member_function::kind() noexcept
{
    return cpp_entity_kind::member_function_t;
}

cpp_entity_kind cpp_member_function::do_get_entity_kind() const noexcept
{
    return kind();
}

cpp_entity_kind cpp_conversion_op::kind() noexcept
{
    return cpp_entity_kind::conversion_op_t;
}

cpp_entity_kind cpp_conversion_op::do_get_entity_kind() const noexcept
{
    return kind();
}

cpp_entity_kind cpp_constructor::kind() noexcept
{
    return cpp_entity_kind::constructor_t;
}

cpp_entity_kind cpp_constructor::do_get_entity_kind() const noexcept
{
    return kind();
}

cpp_entity_kind cpp_destructor::kind() noexcept
{
    return cpp_entity_kind::destructor_t;
}

cpp_entity_kind cpp_destructor::do_get_entity_kind() const noexcept
{
    return kind();
}
