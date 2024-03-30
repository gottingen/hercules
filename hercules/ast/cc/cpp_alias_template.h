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

#include <hercules/ast/cc/cpp_template.h>
#include <hercules/ast/cc/cpp_type_alias.h>

namespace hercules::ccast {
    /// A [hercules::ccast::cpp_entity]() modelling a C++ alias template.
    class cpp_alias_template final : public cpp_template {
    public:
        static cpp_entity_kind kind() noexcept;

        /// Builder for [hercules::ccast::cpp_alias_template]().
        class builder : public basic_builder<cpp_alias_template, cpp_type_alias> {
        public:
            using basic_builder::basic_builder;
        };

        /// \returns A reference to the type alias that is being templated.
        const cpp_type_alias &type_alias() const noexcept {
            return static_cast<const cpp_type_alias &>(*begin());
        }

    private:
        cpp_alias_template(std::unique_ptr<cpp_type_alias> alias)
                : cpp_template(std::unique_ptr<cpp_entity>(alias.release())) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        friend basic_builder<cpp_alias_template, cpp_type_alias>;
    };
} // namespace hercules::ccast