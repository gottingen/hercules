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

#include <collie/type_safe/optional.h>

#include <hercules/ast/cc/cpp_entity.h>
#include <hercules/ast/cc/cpp_entity_container.h>
#include <hercules/ast/cc/cpp_type.h>

namespace hercules::ccast {
    /// A [hercules::ccast::cpp_entity]() representing a friend declaration.
    ///
    /// It can either declare or define a `friend` function (template), declare a `friend` class,
    /// or refer to an existing type.
    class cpp_friend : public cpp_entity, private cpp_entity_container<cpp_friend, cpp_entity> {
    public:
        static cpp_entity_kind kind() noexcept;

        /// \returns A newly created friend declaring the given entity as `friend`.
        /// \notes The friend declaration itself will not be registered,
        /// but the referring entity is.
        static std::unique_ptr<cpp_friend> build(std::unique_ptr<cpp_entity> e) {
            return std::unique_ptr<cpp_friend>(new cpp_friend(std::move(e)));
        }

        /// \returns A newly created friend declaring the given type as `friend`.
        /// \notes It will not be registered.
        static std::unique_ptr<cpp_friend> build(std::unique_ptr<cpp_type> type) {
            return std::unique_ptr<cpp_friend>(new cpp_friend(std::move(type)));
        }

        /// \returns An optional reference to the entity it declares as friend, or `nullptr`.
        collie::ts::optional_ref<const cpp_entity> entity() const noexcept {
            if (begin() == end())
                return nullptr;
            return collie::ts::ref(*begin());
        }

        /// \returns An optional reference to the type it declares as friend, or `nullptr`.
        collie::ts::optional_ref<const cpp_type> type() const noexcept {
            return collie::ts::opt_ref(type_.get());
        }

    private:
        cpp_friend(std::unique_ptr<cpp_entity> e) : cpp_entity("") {
            add_child(std::move(e));
        }

        cpp_friend(std::unique_ptr<cpp_type> type) : cpp_entity(""), type_(std::move(type)) {}

        cpp_entity_kind do_get_entity_kind() const noexcept override;

        std::unique_ptr<cpp_type> type_;

        friend cpp_entity_container<cpp_friend, cpp_entity>;
    };
} // namespace hercules::ccast
