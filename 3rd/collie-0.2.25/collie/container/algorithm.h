// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
//
// Created by jeff on 24-1-10.
//

#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>

namespace collie::container_internal {

    template <typename T, typename SFINAE = void>
    struct MapKeyType : std::false_type {};

    template <typename T>
    struct MapKeyType<T, std::void_t<typename T::key_type>> : std::true_type {};

}  // namespace turbo::container_internal

namespace collie {

    template <typename Container, typename Value, typename std::enable_if_t<container_internal::MapKeyType<Container>::value, int> =0>
    constexpr bool contains(const Container& container, const Value& value) {
        return container.find(value) != container.end();
    }

    // Overload that allows to provide an additional projection invocable. This
    // projection will be applied to every element in `container` before comparing
    // it with `value`. This will always perform a linear search.
    template <typename Container, typename Value, typename std::enable_if_t<!container_internal::MapKeyType<Container>::value, int> =0>
    constexpr bool contains(const Container& container,
                            const Value& value, bool sorted = false) {
        if(sorted) {
            return std::binary_search(container.begin(), container.end(), value);
        }
        return std::find(container.begin(), container.end(), value) != container.end();
    }
}  // namespace collie
