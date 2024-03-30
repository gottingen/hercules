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

#include <hercules/ast/cc/cpp_variable.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

namespace hercules::ccast {

    cpp_entity_kind cpp_variable::kind() noexcept {
        return cpp_entity_kind::variable_t;
    }

    std::unique_ptr<cpp_variable> cpp_variable::build(
            const cpp_entity_index &idx, cpp_entity_id id, std::string name, std::unique_ptr<cpp_type> type,
            std::unique_ptr<cpp_expression> def, cpp_storage_class_specifiers spec, bool is_constexpr,
            collie::ts::optional<cpp_entity_ref> semantic_parent) {
        auto result = std::unique_ptr<cpp_variable>(
                new cpp_variable(std::move(name), std::move(type), std::move(def), spec, is_constexpr));
        result->set_semantic_parent(std::move(semantic_parent));
        idx.register_definition(std::move(id), collie::ts::cref(*result));
        return result;
    }

    std::unique_ptr<cpp_variable> cpp_variable::build_declaration(
            cpp_entity_id definition_id, std::string name, std::unique_ptr<cpp_type> type,
            cpp_storage_class_specifiers spec, bool is_constexpr,
            collie::ts::optional<cpp_entity_ref> semantic_parent) {
        auto result = std::unique_ptr<cpp_variable>(
                new cpp_variable(std::move(name), std::move(type), nullptr, spec, is_constexpr));
        result->set_semantic_parent(std::move(semantic_parent));
        result->mark_declaration(definition_id);
        return result;
    }

    cpp_entity_kind cpp_variable::do_get_entity_kind() const noexcept {
        return kind();
    }
}  // namespace hercules::ccast
