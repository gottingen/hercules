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


#include <hercules/ast/cc/cpp_function.h>

#include <hercules/ast/cc/cpp_entity_kind.h>

namespace hercules::ccast {

    cpp_entity_kind cpp_function_parameter::kind() noexcept {
        return cpp_entity_kind::function_parameter_t;
    }

    std::unique_ptr<cpp_function_parameter> cpp_function_parameter::build(
            const cpp_entity_index &idx, cpp_entity_id id, std::string name, std::unique_ptr<cpp_type> type,
            std::unique_ptr<cpp_expression> def) {
        auto result = std::unique_ptr<cpp_function_parameter>(
                new cpp_function_parameter(std::move(name), std::move(type), std::move(def)));
        idx.register_definition(std::move(id), collie::ts::cref(*result));
        return result;
    }

    std::unique_ptr<cpp_function_parameter> cpp_function_parameter::build(
            std::unique_ptr<cpp_type> type, std::unique_ptr<cpp_expression> def) {
        return std::unique_ptr<cpp_function_parameter>(
                new cpp_function_parameter("", std::move(type), std::move(def)));
    }

    cpp_entity_kind cpp_function_parameter::do_get_entity_kind() const noexcept {
        return kind();
    }

    std::string cpp_function_base::do_get_signature() const {
        std::string result = "(";
        for (auto &param: parameters())
            result += to_string(param.type()) + ',';
        if (is_variadic())
            result += "...";

        if (result.back() == ',')
            result.back() = ')';
        else
            result.push_back(')');

        return result;
    }

    cpp_entity_kind cpp_function::kind() noexcept {
        return cpp_entity_kind::function_t;
    }

    cpp_entity_kind cpp_function::do_get_entity_kind() const noexcept {
        return kind();
    }
}  // namespace hercules::ccast