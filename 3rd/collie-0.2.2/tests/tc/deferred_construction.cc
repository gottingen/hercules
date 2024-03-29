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
#include <collie/type_safe/deferred_construction.h>

#include <collie/testing/doctest.h>
#include <string>

using namespace collie::ts;

TEST_CASE("deferred_construction")
{
    SUBCASE("constructor - empty")
    {
        deferred_construction<int> a;
        REQUIRE(!a.has_value());
    }
    SUBCASE("constructor - copy/move")
    {
        deferred_construction<int> a;
        a = 0;

        deferred_construction<int> b(a);
        REQUIRE(b.has_value());
        REQUIRE(b.value() == 0);

        deferred_construction<int> c(std::move(a));
        REQUIRE(c.has_value());
        REQUIRE(c.value() == 0);

        deferred_construction<int> d;

        deferred_construction<int> e(d);
        REQUIRE(!e.has_value());

        deferred_construction<int> f(std::move(d));
        REQUIRE(!d.has_value());
    }
    SUBCASE("assignment")
    {
        deferred_construction<int> a;
        REQUIRE(!a.has_value());

        a = 42;
        REQUIRE(a.has_value());
        REQUIRE(a.value() == 42);
    }
    SUBCASE("emplace")
    {
        deferred_construction<std::string> a;
        REQUIRE(!a.has_value());

        a.emplace(3u, 'c');
        REQUIRE(a.has_value());
        REQUIRE(a.value() == "ccc");
    }
    SUBCASE("operator bool")
    {
        deferred_construction<int> a;
        REQUIRE(!a);
        a = 42;
        REQUIRE(a);
    }
}
