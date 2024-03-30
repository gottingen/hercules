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

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_expression.h>
#include <hercules/ast/cc/cpp_template_parameter.h>

namespace hercules::ccast {
    /// A [[hercules::ccast::cpp_entity]() modelling a c++ concept declaration
    /// \notes while concepts are technically templates,
    /// this is not a [hercules::ccast::cpp_template](),
    /// as concepts act very differently from other templates
    class cpp_concept final : public cpp_entity {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns the template parameters as a string
        const cpp_token_string &parameters() const noexcept {
            return parameters_;
        }

        /// \returns the [hercules::ccast::cpp_expression]() defining the concept constraint
        const cpp_expression &constraint_expression() const noexcept {
            return *expression_;
        }

        class builder {
        public:
            builder(std::string name) : concept_(new cpp_concept(std::move(name))) {}

            cpp_token_string &set_parameters(cpp_token_string string) noexcept {
                concept_->parameters_ = std::move(string);
                return concept_->parameters_;
            }

            cpp_expression &set_expression(std::unique_ptr<cpp_expression> expression) noexcept {
                concept_->expression_ = std::move(expression);
                return *concept_->expression_;
            }

            std::unique_ptr<cpp_concept> finish(const cpp_entity_index &idx, cpp_entity_id id);

        private:
            std::unique_ptr<cpp_concept> concept_;
        };

    private:
        cpp_concept(std::string name) : cpp_entity(std::move(name)), parameters_({}) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        cpp_token_string parameters_;

        std::unique_ptr<cpp_expression> expression_;
    };

} // namespace hercules::ccast
