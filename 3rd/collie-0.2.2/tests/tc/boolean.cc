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
#include <collie/type_safe/boolean.h>

#include <collie/testing/doctest.h>

#include <sstream>

using namespace collie::ts;

#ifndef TYPE_SAFE_TEST_NO_STATIC_ASSERT
static_assert(std::is_standard_layout<boolean>::value, "");
static_assert(std::is_trivially_copyable<boolean>::value, "");
// conversion checks
static_assert(std::is_constructible<boolean, bool>::value, "");
static_assert(!std::is_constructible<boolean, int>::value, "");
static_assert(std::is_assignable<boolean, bool>::value, "");
static_assert(!std::is_assignable<boolean, int>::value, "");
#endif

TEST_CASE("boolean")
{
    SUBCASE("constructor")
    {
        boolean b1(true);
        REQUIRE(static_cast<bool>(b1));

        boolean b2(false);
        REQUIRE(!static_cast<bool>(b2));
    }
    SUBCASE("assignment")
    {
        boolean b1(true);
        b1 = false;
        REQUIRE(!static_cast<bool>(b1));
        b1 = true;
        REQUIRE(static_cast<bool>(b1));
    }
    SUBCASE("negate")
    {
        boolean b1(true);
        REQUIRE(!b1 == false);

        boolean b2(false);
        REQUIRE(!b2 == true);
    }
    SUBCASE("comparison")
    {
        boolean b1(true);
        REQUIRE(b1 == true);
        REQUIRE(true == b1);
        REQUIRE(b1 != false);
        REQUIRE(false != b1);
        REQUIRE(b1 == boolean(true));
        REQUIRE(b1 != boolean(false));

        boolean b2(false);
        REQUIRE(b2 == false);
        REQUIRE(false == b2);
        REQUIRE(b2 != true);
        REQUIRE(true != b2);
        REQUIRE(b2 == boolean(false));
        REQUIRE(b2 != boolean(true));
    }
    SUBCASE("i/o")
    {
        std::ostringstream out;
        std::istringstream in("0");

        boolean b(true);
        out << b;
        REQUIRE(out.str() == "1");

        in >> b;
        REQUIRE(!static_cast<bool>(b));
    }
}
