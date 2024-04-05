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

#include <utility>

namespace collie::ts::detail {
    template<typename Func, typename Value, typename... Args>
    auto map_invoke(Func &&f, Value &&v, Args &&... args)
    -> decltype(std::forward<Func>(f)(std::forward<Value>(v), std::forward<Args>(args)...)) {
        return std::forward<Func>(f)(std::forward<Value>(v), std::forward<Args>(args)...);
    }

    template<typename Func, typename Value>
    auto map_invoke(Func &&f, Value &&v) -> decltype(std::forward<Value>(v).*std::forward<Func>(f)) {
        return std::forward<Value>(v).*std::forward<Func>(f);
    }

    template<typename Func, typename Value, typename... Args>
    auto map_invoke(Func &&f, Value &&v, Args &&... args)
    -> decltype((std::forward<Value>(v).*std::forward<Func>(f))(std::forward<Args>(args)...)) {
        return (std::forward<Value>(v).*std::forward<Func>(f))(std::forward<Args>(args)...);
    }
} // namespace collie::ts::detail
