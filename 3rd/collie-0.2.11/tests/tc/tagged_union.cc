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
#include <collie/type_safe/tagged_union.h>

#include <collie/testing/doctest.h>

#include "debugger_type.hpp"

using namespace collie::ts;

TEST_CASE("tagged_union")
{
    using union_t = tagged_union<int, double, debugger_type>;
    REQUIRE(union_t::invalid_type == union_t::type_id());

    union_t tunion;
    REQUIRE(!tunion.has_value());
    REQUIRE(tunion.type() == union_t::invalid_type);
    REQUIRE(static_cast<std::size_t>(tunion.type()) == 0u);

    SUBCASE("emplace int")
    {
        tunion.emplace(union_type<int>{}, 5);
        REQUIRE(tunion.has_value());
        REQUIRE(tunion.type() == union_t::type_id(union_type<int>{}));
        REQUIRE(static_cast<std::size_t>(tunion.type()) == 1u);
        REQUIRE(tunion.value(union_type<int>{}) == 5);

        union_t other;
        copy(other, tunion);
        REQUIRE(other.has_value());
        REQUIRE(other.type() == union_t::type_id(union_type<int>{}));
        REQUIRE(other.value(union_type<int>{}) == 5);

        SUBCASE("member")
        {
            tunion.destroy(union_type<int>{});
            REQUIRE(!tunion.has_value());
            REQUIRE(tunion.type() == union_t::invalid_type);
        }
        SUBCASE("non-member")
        {
            destroy(tunion);
            REQUIRE(!tunion.has_value());
            REQUIRE(tunion.type() == union_t::invalid_type);
        }
    }
    SUBCASE("emplace double")
    {
        tunion.emplace(union_type<double>{}, 3.0);
        REQUIRE(tunion.has_value());
        REQUIRE(tunion.type() == union_t::type_id(union_type<double>{}));
        REQUIRE(static_cast<std::size_t>(tunion.type()) == 2u);
        REQUIRE(tunion.value(union_type<double>{}) == 3.0);

        union_t other;
        copy(other, tunion);
        REQUIRE(other.has_value());
        REQUIRE(other.type() == union_t::type_id(union_type<double>{}));
        REQUIRE(other.value(union_type<double>{}) == 3.0);

        SUBCASE("member")
        {
            tunion.destroy(union_type<double>{});
            REQUIRE(!tunion.has_value());
            REQUIRE(tunion.type() == union_t::invalid_type);
        }
        SUBCASE("non-member")
        {
            destroy(tunion);
            REQUIRE(!tunion.has_value());
            REQUIRE(tunion.type() == union_t::invalid_type);
        }
    }
    SUBCASE("emplace debugger_type")
    {
        tunion.emplace(union_type<debugger_type>{}, 42);
        REQUIRE(tunion.has_value());
        REQUIRE(tunion.type() == union_t::type_id(union_type<debugger_type>{}));
        REQUIRE(static_cast<std::size_t>(tunion.type()) == 3u);

        auto& val = tunion.value(union_type<debugger_type>{});
        REQUIRE(val.id == 42);
        REQUIRE(val.ctor());

        SUBCASE("copy")
        {
            union_t other;
            copy(other, tunion);
            REQUIRE(other.has_value());
            REQUIRE(other.type() == union_t::type_id(union_type<debugger_type>{}));

            auto& otherval = other.value(union_type<debugger_type>{});
            REQUIRE(otherval.id == 42);
            REQUIRE(otherval.copy_ctor());
        }
        SUBCASE("move")
        {
            union_t other;
            move(other, std::move(tunion));
            REQUIRE(other.has_value());
            REQUIRE(other.type() == union_t::type_id(union_type<debugger_type>{}));

            auto& otherval = other.value(union_type<debugger_type>{});
            REQUIRE(otherval.id == 42);
            REQUIRE(otherval.move_ctor());
        }

        SUBCASE("member")
        {
            tunion.destroy(union_type<debugger_type>{});
            REQUIRE(!tunion.has_value());
            REQUIRE(tunion.type() == union_t::invalid_type);
        }
        SUBCASE("non-member")
        {
            destroy(tunion);
            REQUIRE(!tunion.has_value());
            REQUIRE(tunion.type() == union_t::invalid_type);
        }
    }
}
