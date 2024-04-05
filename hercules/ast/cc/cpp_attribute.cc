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


#include <hercules/ast/cc/cpp_attribute.h>

#include <algorithm>

#include <hercules/ast/cc/cpp_entity.h>

namespace hercules::ccast {

    namespace {
        const char *get_attribute_name(cpp_attribute_kind kind) noexcept {
            switch (kind) {
                case cpp_attribute_kind::alignas_:
                    return "alignas";
                case cpp_attribute_kind::carries_dependency:
                    return "carries_dependency";
                case cpp_attribute_kind::deprecated:
                    return "deprecated";
                case cpp_attribute_kind::fallthrough:
                    return "fallthrough";
                case cpp_attribute_kind::maybe_unused:
                    return "maybe_unused";
                case cpp_attribute_kind::nodiscard:
                    return "nodiscard";
                case cpp_attribute_kind::noreturn:
                    return "noreturn";

                case cpp_attribute_kind::unknown:
                    return "unknown";
            }

            return "<error>";
        }
    } // namespace

    cpp_attribute::cpp_attribute(cpp_attribute_kind kind,
                                 collie::ts::optional<cpp_token_string> arguments)
            : cpp_attribute(collie::ts::nullopt, get_attribute_name(kind), std::move(arguments), false) {
        kind_ = kind;
    }

    collie::ts::optional_ref<const cpp_attribute> has_attribute(
            const cpp_attribute_list &attributes, const std::string &name) {
        auto iter
                = std::find_if(attributes.begin(), attributes.end(), [&](const cpp_attribute &attribute) {
                    if (attribute.scope())
                        return attribute.scope().value() + "::" + attribute.name() == name;
                    else
                        return attribute.name() == name;
                });

        if (iter == attributes.end())
            return nullptr;
        else
            return collie::ts::ref(*iter);
    }

    collie::ts::optional_ref<const cpp_attribute> has_attribute(
            const cpp_attribute_list &attributes, cpp_attribute_kind kind) {
        auto iter
                = std::find_if(attributes.begin(), attributes.end(),
                               [&](const cpp_attribute &attribute) { return attribute.kind() == kind; });

        if (iter == attributes.end())
            return nullptr;
        else
            return collie::ts::ref(*iter);
    }

    collie::ts::optional_ref<const cpp_attribute> has_attribute(const cpp_entity &e,
                                                                                 const std::string &name) {
        return has_attribute(e.attributes(), name);
    }

    collie::ts::optional_ref<const cpp_attribute> has_attribute(const cpp_entity &e,
                                                                                 cpp_attribute_kind kind) {
        return has_attribute(e.attributes(), kind);
    }
}  // namespace hercules::ccast
