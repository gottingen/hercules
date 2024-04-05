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
#include <hercules/ast/cc/cpp_entity_container.h>

namespace hercules::ccast {
    /// A [hercules::ccast::cpp_entity]() modelling a language linkage.
    class cpp_language_linkage final : public cpp_entity,
                                       public cpp_entity_container<cpp_language_linkage, cpp_entity> {
    public:
        static cpp_entity_kind kind() noexcept;

        /// Builds a [hercules::ccast::cpp_language_linkage]().
        class builder {
        public:
            /// \effects Sets the name, that is the kind of language linkage.
            explicit builder(std::string name) : linkage_(new cpp_language_linkage(std::move(name))) {}

            /// \effects Adds an entity to the language linkage.
            void add_child(std::unique_ptr<cpp_entity> child) {
                linkage_->add_child(std::move(child));
            }

            /// \returns The not yet finished language linkage.
            cpp_language_linkage &get() const noexcept {
                return *linkage_;
            }

            /// \returns The finalized language linkage.
            /// \notes It is not registered on purpose as nothing can refer to it.
            std::unique_ptr<cpp_language_linkage> finish() {
                return std::move(linkage_);
            }

        private:
            std::unique_ptr<cpp_language_linkage> linkage_;
        };

        /// \returns `true` if the linkage is a block, `false` otherwise.
        bool is_block() const noexcept;

    private:
        using cpp_entity::cpp_entity;

        cpp_entity_kind do_get_entity_kind() const noexcept override;
    };
} // namespace hercules::ccast
