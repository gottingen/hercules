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
#include <collie/type_safe/downcast.h>

#include <collie/testing/doctest.h>

using namespace collie::ts;

TEST_CASE("downcast")
{
    struct base
    {
        virtual ~base() = default;
    };

    struct derived : base
    {
        ~derived() override = default;
    };

    base    b;
    derived d;

    SUBCASE("base -> base")
    {
        base& ref = b;

        base& res1 = downcast<base&>(ref);
        REQUIRE(&res1 == &ref);

        base& res2 = downcast(derived_type<base>{}, ref);
        REQUIRE(&res2 == &ref);
    }
    SUBCASE("const base -> const base")
    {
        const base& ref = b;

        const base& res1 = downcast<const base&>(ref);
        REQUIRE(&res1 == &ref);

        const base& res2 = downcast(derived_type<base>{}, ref);
        REQUIRE(&res2 == &ref);
    }
    SUBCASE("base -> derived")
    {
        base& ref = d;

        derived& res1 = downcast<derived&>(ref);
        REQUIRE(&res1 == &ref);

        derived& res2 = downcast(derived_type<derived>{}, ref);
        REQUIRE(&res2 == &ref);
    }
    SUBCASE("const base -> const derived")
    {
        const base& ref = d;

        const derived& res1 = downcast<const derived&>(ref);
        REQUIRE(&res1 == &ref);

        const derived& res2 = downcast(derived_type<derived>{}, ref);
        REQUIRE(&res2 == &ref);
    }
}
