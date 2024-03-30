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

namespace collie::ts::detail {
    template<bool AllowCopy>
    struct copy_control;

    template<>
    struct copy_control<true> {
        copy_control() noexcept = default;

        copy_control(const copy_control &) noexcept = default;

        copy_control &operator=(const copy_control &) noexcept = default;

        copy_control(copy_control &&) noexcept = default;

        copy_control &operator=(copy_control &&) noexcept = default;
    };

    template<>
    struct copy_control<false> {
        copy_control() noexcept = default;

        copy_control(const copy_control &) noexcept = delete;

        copy_control &operator=(const copy_control &) noexcept = delete;

        copy_control(copy_control &&) noexcept = default;

        copy_control &operator=(copy_control &&) noexcept = default;
    };

    template<bool AllowCopy>
    struct move_control;

    template<>
    struct move_control<true> {
        move_control() noexcept = default;

        move_control(const move_control &) noexcept = default;

        move_control &operator=(const move_control &) noexcept = default;

        move_control(move_control &&) noexcept = default;

        move_control &operator=(move_control &&) noexcept = default;
    };

    template<>
    struct move_control<false> {
        move_control() noexcept = default;

        move_control(const move_control &) noexcept = default;

        move_control &operator=(const move_control &) noexcept = default;

        move_control(move_control &&) noexcept = delete;

        move_control &operator=(move_control &&) noexcept = delete;
    };
} // namespace collie::ts::detail
