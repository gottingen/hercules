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
// Created by jeff on 24-1-5.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

#include "turbo/base/check_math.h"

TEST_CASE("NextPow2") {

    static_assert(turbo::next_pow2(0u) == 1);
    static_assert(turbo::next_pow2(1u) == 1);
    static_assert(turbo::next_pow2(100u) == 128u);
    static_assert(turbo::next_pow2(245u) == 256u);
    static_assert(turbo::next_pow2(512u) == 512u);
    static_assert(turbo::next_pow2(513u) == 1024u);

    REQUIRE(turbo::next_pow2(0u) == 1u);
    REQUIRE(turbo::next_pow2(2u) == 2u);
    REQUIRE(turbo::next_pow2(1u) == 1u);
    REQUIRE(turbo::next_pow2(33u) == 64u);
    REQUIRE(turbo::next_pow2(100u) == 128u);
    REQUIRE(turbo::next_pow2(211u) == 256u);
    REQUIRE(turbo::next_pow2(23u) == 32u);
    REQUIRE(turbo::next_pow2(54u) == 64u);

    uint64_t z = 0;
    uint64_t a = 1;
    REQUIRE(turbo::next_pow2(z) == 1);
    REQUIRE(turbo::next_pow2(a) == a);
    REQUIRE(turbo::next_pow2((a << 5) + 0) == (a << 5));
    REQUIRE(turbo::next_pow2((a << 5) + 1) == (a << 6));
    REQUIRE(turbo::next_pow2((a << 32) + 0) == (a << 32));
    REQUIRE(turbo::next_pow2((a << 32) + 1) == (a << 33));
    REQUIRE(turbo::next_pow2((a << 41) + 0) == (a << 41));
    REQUIRE(turbo::next_pow2((a << 41) + 1) == (a << 42));

    REQUIRE(turbo::is_pow2(0) == false);
    REQUIRE(turbo::is_pow2(1) == true);
    REQUIRE(turbo::is_pow2(2) == true);
    REQUIRE(turbo::is_pow2(3) == false);
    REQUIRE(turbo::is_pow2(0u) == false);
    REQUIRE(turbo::is_pow2(1u) == true);
    REQUIRE(turbo::is_pow2(54u) == false);
    REQUIRE(turbo::is_pow2(64u) == true);
}
