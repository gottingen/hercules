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
#include <collie/type_safe/index.h>

#include <collie/testing/doctest.h>

using collie::ts::difference_t;
using collie::ts::index_t;

TEST_CASE("index_t")
{
    index_t idx;
    REQUIRE(idx == index_t(0u));

    SUBCASE("operator+=")
    {
        difference_t a(5);
        idx += a;
        REQUIRE(idx == index_t(5u));

        difference_t b(-3);
        idx += b;
        REQUIRE(idx == index_t(2u));
    }
    SUBCASE("operator-=")
    {
        idx = index_t(10u);

        difference_t a(5);
        idx -= a;
        REQUIRE(idx == index_t(5u));

        difference_t b(-3);
        idx -= b;
        REQUIRE(idx == index_t(8u));
    }
    SUBCASE("operator+")
    {
        auto c = idx + difference_t(5);
        REQUIRE(c == index_t(5u));

        auto d = difference_t(5) + idx;
        REQUIRE(d == index_t(5u));

        auto e = c + difference_t(-3);
        REQUIRE(e == index_t(2u));

        auto f = difference_t(-3) + d;
        REQUIRE(f == index_t(2u));
    }
    SUBCASE("next")
    {
        auto a = next(idx, difference_t(5));
        REQUIRE(a == index_t(5u));

        auto b = next(a, difference_t(-3));
        REQUIRE(b == index_t(2u));
    }
    SUBCASE("prev")
    {
        idx = index_t(10u);

        auto a = prev(idx, difference_t(5));
        REQUIRE(a == index_t(5u));

        auto b = prev(a, difference_t(-3));
        REQUIRE(b == index_t(8u));
    }
    SUBCASE("advance")
    {
        advance(idx, difference_t(5));
        REQUIRE(idx == index_t(5u));

        advance(idx, difference_t(-3));
        REQUIRE(idx == index_t(2u));
    }
    SUBCASE("operator-")
    {
        idx = index_t(10u);

        auto c = idx - difference_t(5);
        REQUIRE(c == index_t(5u));

        auto d = c - difference_t(-3);
        REQUIRE(d == index_t(8u));
    }
    SUBCASE("distance")
    {
        auto a = index_t(5u) - idx;
        REQUIRE(a == difference_t(5));
        REQUIRE(a == distance(idx, index_t(5u)));

        auto b = idx - index_t(5u);
        REQUIRE(b == difference_t(-5));
        REQUIRE(b == distance(index_t(5u), idx));
    }
    SUBCASE("at")
    {
        std::size_t array[] = {0, 1, 2, 3, 4, 5};

        for (index_t i; i != 5u; ++i)
            REQUIRE(at(array, i) == std::size_t(get(i)));
    }
}
