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

#include <hercules/ast/cc/cpp_function.h>
#include <hercules/ast/cc/cpp_template.h>

namespace hercules::ccast {
    /// A [hercules::ccast::cpp_entity]() modelling a function template.
    class cpp_function_template final : public cpp_template {
    public:
        static cpp_entity_kind kind() noexcept;

        /// Builder for [hercules::ccast::cpp_function_template]().
        class builder : public basic_builder<cpp_function_template, cpp_function_base> {
        public:
            using basic_builder::basic_builder;
        };

        /// A reference to the function that is being templated.
        const cpp_function_base &function() const noexcept {
            return static_cast<const cpp_function_base &>(*begin());
        }

    private:
        cpp_function_template(std::unique_ptr<cpp_function_base> func)
                : cpp_template(std::unique_ptr<cpp_entity>(func.release())) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        friend basic_builder<cpp_function_template, cpp_function_base>;
    };

    /// A [hercules::ccast::cpp_entity]() modelling a function template specialization.
    class cpp_function_template_specialization final : public cpp_template_specialization {
    public:
        static cpp_entity_kind kind() noexcept;

        /// Builder for [hercules::ccast::cpp_function_template_specialization]().
        class builder
                : public specialization_builder<cpp_function_template_specialization, cpp_function_base> {
        public:
            using specialization_builder::specialization_builder;

        private:
            using specialization_builder::add_parameter;
        };

        /// A reference to the function that is being specialized.
        const cpp_function_base &function() const noexcept {
            return static_cast<const cpp_function_base &>(*begin());
        }

    private:
        cpp_function_template_specialization(std::unique_ptr<cpp_function_base> func,
                                             cpp_template_ref primary)
                : cpp_template_specialization(std::unique_ptr<cpp_entity>(func.release()), primary) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        friend specialization_builder<cpp_function_template_specialization, cpp_function_base>;
    };
} // namespace hercules::ccast
