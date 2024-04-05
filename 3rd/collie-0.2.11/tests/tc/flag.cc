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
#include <collie/type_safe/flag.h>

#include <collie/testing/doctest.h>

using namespace collie::ts;

TEST_CASE("flag")
{
    SUBCASE("constructor")
    {
        flag a(true);
        REQUIRE(a == true);

        flag b(false);
        REQUIRE(b == false);
    }
    SUBCASE("toggle")
    {
        flag a(true);
        REQUIRE(a.toggle());
        REQUIRE(a == false);

        flag b(false);
        REQUIRE(!b.toggle());
        REQUIRE(b == true);
    }
    SUBCASE("change")
    {
        flag a(true);
        a.change(false);
        REQUIRE(a == false);

        flag b(false);
        b.change(true);
        REQUIRE(b == true);
    }
    SUBCASE("set")
    {
        flag a(true);
        a.set();
        REQUIRE(a == true);

        flag b(false);
        b.set();
        REQUIRE(b == true);
    }
    SUBCASE("try_set")
    {
        flag a(true);
        REQUIRE(!a.try_set());
        REQUIRE(a == true);

        flag b(false);
        REQUIRE(b.try_set());
        REQUIRE(b == true);
    }
    SUBCASE("reset")
    {
        flag a(true);
        a.reset();
        REQUIRE(a == false);

        flag b(false);
        b.reset();
        REQUIRE(b == false);
    }
    SUBCASE("try_reset")
    {
        flag a(true);
        REQUIRE(a.try_reset());
        REQUIRE(a == false);

        flag b(false);
        REQUIRE(!b.try_reset());
        REQUIRE(b == false);
    }
}
