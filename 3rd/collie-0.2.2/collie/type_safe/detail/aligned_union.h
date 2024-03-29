// Copyright 2024 The Elastic-AI Authors.
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

#pragma once

#include <cstddef>
#include <type_traits>

namespace collie::ts::detail {
    // max for variadic number of types.
    template<typename T>
    constexpr const T &max(const T &a) {
        return a;
    }

    template<typename T>
    constexpr const T &max(const T &a, const T &b) {
        return a < b ? b : a;
    }

    template<typename T, typename... Ts>
    constexpr const T &max(const T &t, const Ts &... ts) {
        return max(t, max(ts...));
    }

    template<typename... Types>
    class aligned_union {
    public:
        static constexpr auto size_value = detail::max(sizeof(Types)...);
        static constexpr auto alignment_value = detail::max(alignof(Types)...);

        void *get() noexcept {
            return &storage_;
        }

        const void *get() const noexcept {
            return &storage_;
        }

    private:
        alignas(alignment_value) unsigned char storage_[size_value];
    };
} // namespace collie::ts::detail

