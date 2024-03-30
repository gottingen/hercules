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

#include <hercules/ast/cc/cpp_type_alias.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

namespace hercules::ccast {

    cpp_entity_kind cpp_type_alias::kind() noexcept {
        return cpp_entity_kind::type_alias_t;
    }

    std::unique_ptr<cpp_type_alias> cpp_type_alias::build(const cpp_entity_index &idx, cpp_entity_id id,
                                                          std::string name,
                                                          std::unique_ptr<cpp_type> type) {
        auto result = build(std::move(name), std::move(type));
        idx.register_forward_declaration(std::move(id), collie::ts::cref(*result)); // not a definition
        return result;
    }

    std::unique_ptr<cpp_type_alias> cpp_type_alias::build(std::string name,
                                                          std::unique_ptr<cpp_type> type) {
        return std::unique_ptr<cpp_type_alias>(new cpp_type_alias(std::move(name), std::move(type)));
    }

    cpp_entity_kind cpp_type_alias::do_get_entity_kind() const noexcept {
        return kind();
    }
}  // namespace hercules::ccast