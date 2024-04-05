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

#include <hercules/ast/cc/cpp_template_parameter.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

namespace hercules::ccast {

    const char *to_string(cpp_template_keyword kw) noexcept {
        switch (kw) {
            case cpp_template_keyword::keyword_class:
                return "class";
            case cpp_template_keyword::keyword_typename:
                return "typename";
            case cpp_template_keyword::concept_contraint:
                return "concept constraint data lost";
        }

        return "should not get here";
    }

    std::unique_ptr<cpp_template_type_parameter> cpp_template_type_parameter::build(
            const cpp_entity_index &idx, cpp_entity_id id, std::string name, cpp_template_keyword kw,
            bool variadic, std::unique_ptr<cpp_type> default_type,
            collie::ts::optional<cpp_token_string> concept_constraint) {
        std::unique_ptr<cpp_template_type_parameter> result(
                new cpp_template_type_parameter(std::move(name), kw, variadic, std::move(default_type),
                                                std::move(concept_constraint)));
        idx.register_definition(std::move(id), collie::ts::cref(*result));
        return result;
    }

    cpp_entity_kind cpp_template_type_parameter::kind() noexcept {
        return cpp_entity_kind::template_type_parameter_t;
    }

    cpp_entity_kind cpp_template_type_parameter::do_get_entity_kind() const noexcept {
        return kind();
    }

    bool detail::cpp_template_parameter_ref_predicate::operator()(const cpp_entity &e) {
        return e.kind() == cpp_entity_kind::template_type_parameter_t;
    }

    std::unique_ptr<cpp_non_type_template_parameter> cpp_non_type_template_parameter::build(
            const cpp_entity_index &idx, cpp_entity_id id, std::string name, std::unique_ptr<cpp_type> type,
            bool is_variadic, std::unique_ptr<cpp_expression> default_value) {
        std::unique_ptr<cpp_non_type_template_parameter> result(
                new cpp_non_type_template_parameter(std::move(name), std::move(type), is_variadic,
                                                    std::move(default_value)));
        idx.register_definition(std::move(id), collie::ts::cref(*result));
        return result;
    }

    cpp_entity_kind cpp_non_type_template_parameter::kind() noexcept {
        return cpp_entity_kind::non_type_template_parameter_t;
    }

    cpp_entity_kind cpp_non_type_template_parameter::do_get_entity_kind() const noexcept {
        return kind();
    }

    bool detail::cpp_template_ref_predicate::operator()(const cpp_entity &e) {
        return is_template(e.kind()) || e.kind() == cpp_entity_kind::template_template_parameter_t;
    }

    cpp_entity_kind cpp_template_template_parameter::kind() noexcept {
        return cpp_entity_kind::template_template_parameter_t;
    }

    cpp_entity_kind cpp_template_template_parameter::do_get_entity_kind() const noexcept {
        return kind();
    }
}  // namespace hercules::ccast
