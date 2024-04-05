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


#include <hercules/ast/cc/cpp_member_variable.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

namespace hercules::ccast {

    cpp_entity_kind cpp_member_variable::kind() noexcept {
        return cpp_entity_kind::member_variable_t;
    }

    std::unique_ptr<cpp_member_variable> cpp_member_variable::build(const cpp_entity_index &idx,
                                                                    cpp_entity_id id, std::string name,
                                                                    std::unique_ptr<cpp_type> type,
                                                                    std::unique_ptr<cpp_expression> def,
                                                                    bool is_mutable) {
        auto result = std::unique_ptr<cpp_member_variable>(
                new cpp_member_variable(std::move(name), std::move(type), std::move(def), is_mutable));
        idx.register_definition(std::move(id), collie::ts::cref(*result));
        return result;
    }

    cpp_entity_kind cpp_member_variable::do_get_entity_kind() const noexcept {
        return kind();
    }

    cpp_entity_kind cpp_bitfield::kind() noexcept {
        return cpp_entity_kind::bitfield_t;
    }

    std::unique_ptr<cpp_bitfield> cpp_bitfield::build(const cpp_entity_index &idx, cpp_entity_id id,
                                                      std::string name, std::unique_ptr<cpp_type> type,
                                                      unsigned no_bits, bool is_mutable) {
        auto result = std::unique_ptr<cpp_bitfield>(
                new cpp_bitfield(std::move(name), std::move(type), no_bits, is_mutable));
        idx.register_definition(std::move(id), collie::ts::cref(*result));
        return result;
    }

    std::unique_ptr<cpp_bitfield> cpp_bitfield::build(std::unique_ptr<cpp_type> type, unsigned no_bits,
                                                      bool is_mutable) {
        return std::unique_ptr<cpp_bitfield>(
                new cpp_bitfield("", std::move(type), no_bits, is_mutable));
    }

    cpp_entity_kind cpp_bitfield::do_get_entity_kind() const noexcept {
        return kind();
    }
}  // namespace hercules::ccast
