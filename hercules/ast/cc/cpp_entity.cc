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


#include <hercules/ast/cc/cpp_entity.h>

#include <hercules/ast/cc/cpp_entity_index.h>
#include <hercules/ast/cc/cpp_entity_kind.h>
#include <hercules/ast/cc/cpp_template.h>

namespace hercules::ccast {

    cpp_scope_name::cpp_scope_name(collie::ts::object_ref<const cpp_entity> entity) : entity_(entity) {
        if (hercules::ccast::is_templated(*entity)) {
            auto &templ = static_cast<const cpp_template &>(entity->parent().value());
            if (!templ.parameters().empty())
                templ_ = collie::ts::ref(templ);
        } else if (is_template(entity->kind())) {
            auto &templ = static_cast<const cpp_template &>(*entity);
            if (!templ.parameters().empty())
                templ_ = collie::ts::ref(templ);
        }
    }

    const std::string &cpp_scope_name::name() const noexcept {
        return entity_->name();
    }

    detail::iteratable_intrusive_list<cpp_template_parameter> cpp_scope_name::template_parameters()
    const noexcept {
        DEBUG_ASSERT(is_templated(), detail::precondition_error_handler{});
        return templ_.value().parameters();
    }

    cpp_entity_kind cpp_unexposed_entity::kind() noexcept {
        return cpp_entity_kind::unexposed_t;
    }

    std::unique_ptr<cpp_entity> cpp_unexposed_entity::build(const cpp_entity_index &index,
                                                            cpp_entity_id id, std::string name,
                                                            cpp_token_string spelling) {
        std::unique_ptr<cpp_entity> result(
                new cpp_unexposed_entity(std::move(name), std::move(spelling)));
        index.register_forward_declaration(id, collie::ts::ref(*result));
        return result;
    }

    std::unique_ptr<cpp_entity> cpp_unexposed_entity::build(cpp_token_string spelling) {
        return std::unique_ptr<cpp_entity>(new cpp_unexposed_entity("", std::move(spelling)));
    }

    cpp_entity_kind cpp_unexposed_entity::do_get_entity_kind() const noexcept {
        return kind();
    }

    bool is_templated(const cpp_entity &e) noexcept {
        if (!e.parent())
            return false;
        else if (!is_template(e.parent().value().kind()))
            return false;
        return e.parent().value().name() == e.name();
    }
}