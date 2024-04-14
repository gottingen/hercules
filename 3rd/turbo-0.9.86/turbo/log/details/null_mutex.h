// Copyright 2023 The titan-search Authors.
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

#include <atomic>
#include <utility>
// null, no cost dummy "mutex" and dummy "atomic" int

namespace turbo::tlog::details {
    struct null_mutex {
        void lock() const {}

        void unlock() const {}
    };

    struct null_atomic_int {
        int value;

        null_atomic_int() = default;

        explicit null_atomic_int(int new_value)
                : value(new_value) {}

        int load(std::memory_order = std::memory_order_relaxed) const {
            return value;
        }

        void store(int new_value, std::memory_order = std::memory_order_relaxed) {
            value = new_value;
        }

        int exchange(int new_value, std::memory_order = std::memory_order_relaxed) {
            std::swap(new_value, value);
            return new_value; // return value before the call
        }
    };

} // namespace turbo::tlog::details
