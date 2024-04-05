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

namespace hercules::ccast {
    /// Helper class for entities that are containers.
    ///
    /// Inherit from it to generate container access.
    template<class Derived, typename T>
    class cpp_entity_container {
    public:
        using iterator = typename detail::intrusive_list<T>::const_iterator;

        /// \returns A const iterator to the first child.
        iterator begin() const noexcept {
            return children_.begin();
        }

        /// \returns A const iterator to the last child.
        iterator end() const noexcept {
            return children_.end();
        }

    protected:
        /// \effects Adds a new child to the container.
        void add_child(std::unique_ptr<T> ptr) noexcept {
            children_.push_back(static_cast<Derived &>(*this), std::move(ptr));
        }

        /// \returns A non-const iterator to the first child.
        typename detail::intrusive_list<T>::iterator mutable_begin() noexcept {
            return children_.begin();
        }

        /// \returns A non-const iterator one past the last child.
        typename detail::intrusive_list<T>::iterator mutable_end() noexcept {
            return children_.begin();
        }

        ~cpp_entity_container() noexcept = default;

    private:
        detail::intrusive_list<T> children_;
    };
} // namespace hercules::ccast
