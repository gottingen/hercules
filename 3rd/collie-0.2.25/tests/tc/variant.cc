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
#include <collie/type_safe/variant.h>

#include <collie/testing/doctest.h>

#include "debugger_type.hpp"

using collie::ts::fallback_variant;
using collie::ts::nullvar;
using collie::ts::nullvar_t;
using collie::ts::tagged_union;
using collie::ts::union_type;
using collie::ts::variant;
using collie::ts::variant_type;

// use optional variant to be able to test all functions
// test policies separately
using variant_t = variant<nullvar_t, int, double, debugger_type>;
using union_t   = tagged_union<int, double, debugger_type>;

template <class Variant>
void check_variant_empty(const Variant& var)
{
    // boolean queries
    REQUIRE(!var.has_value());
    REQUIRE(!var);

    // type queries
    REQUIRE(var.type() == Variant::invalid_type);
    REQUIRE(var.has_value(variant_type<nullvar_t>{}));

    // value query
    // (check that it compiles and does not abort)
    nullvar_t null = var.value(variant_type<nullvar_t>{});
    (void)null;

    // optional_value queries
    REQUIRE(var.optional_value(variant_type<nullvar_t>{}));
    REQUIRE(!var.optional_value(variant_type<int>{}));
}

template <class Variant, typename T>
void check_variant_value(const Variant& var, const T& val)
{
    // boolean queries
    REQUIRE(var.has_value());
    REQUIRE(var);

    // type queries
    REQUIRE(var.type() == typename Variant::type_id(variant_type<T>{}));
    REQUIRE(var.has_value(variant_type<T>{}));

    // value query
    REQUIRE(var.value(variant_type<T>{}) == val);

    // optional_value queries
    using is_int    = std::is_same<T, int>;
    using is_double = std::is_same<T, double>;
    using is_dbg    = std::is_same<T, debugger_type>;

    REQUIRE(!var.optional_value(variant_type<nullvar_t>{}));
    REQUIRE(bool(var.optional_value(variant_type<int>{})) == is_int::value);
    REQUIRE(bool(var.optional_value(variant_type<double>{})) == is_double::value);
    REQUIRE(bool(var.optional_value(variant_type<debugger_type>{})) == is_dbg::value);
}

TEST_CASE("basic_variant")
{
    variant_t empty;
    variant_t non_empty1(5);
    variant_t non_empty2(debugger_type(42));

    SUBCASE("constructor - empty")
    {
        variant_t a;
        check_variant_empty(a);

        variant_t b(nullvar);
        check_variant_empty(b);
    }
    SUBCASE("constructor - value")
    {
        variant_t a(5);
        check_variant_value(a, 5);

        variant_t b(3.0);
        check_variant_value(b, 3.0);

        variant_t c(debugger_type(42));
        check_variant_value(c, debugger_type(42));
        REQUIRE(c.value(variant_type<debugger_type>{}).move_ctor());
    }
    SUBCASE("constructor - args")
    {
        variant_t a(variant_type<int>{}, 5);
        check_variant_value(a, 5);

        variant_t b(variant_type<double>{}, 3.0);
        check_variant_value(b, 3.0);

        variant_t c(variant_type<debugger_type>{}, 42);
        check_variant_value(c, debugger_type(42));
        REQUIRE(c.value(variant_type<debugger_type>{}).ctor());
    }
    SUBCASE("constructor - union copy")
    {
        union_t u;

        variant_t a(u);
        check_variant_empty(a);

        u.emplace(union_type<int>{}, 5);

        variant_t b(u);
        check_variant_value(b, 5);

        u.destroy(union_type<int>{});
        u.emplace(union_type<debugger_type>{}, debugger_type(42));

        variant_t c(u);
        check_variant_value(c, debugger_type(42));
        REQUIRE(c.value(variant_type<debugger_type>{}).copy_ctor());
    }
    SUBCASE("constructor - union move")
    {
        union_t u;

        variant_t a(std::move(u));
        check_variant_empty(a);

        u.emplace(union_type<int>{}, 5);

        variant_t b(std::move(u));
        check_variant_value(b, 5);

        u.destroy(union_type<int>{});
        u.emplace(union_type<debugger_type>{}, debugger_type(42));

        variant_t c(std::move(u));
        check_variant_value(c, debugger_type(42));
        REQUIRE(c.value(variant_type<debugger_type>{}).move_ctor());
    }
    SUBCASE("constructor - copy")
    {
        variant_t a(empty);
        check_variant_empty(a);

        variant_t b(non_empty1);
        check_variant_value(b, 5);

        variant_t c(non_empty2);
        check_variant_value(c, debugger_type(42));
        REQUIRE(c.value(variant_type<debugger_type>{}).copy_ctor());
    }
    SUBCASE("constructor - move")
    {
        variant_t a(std::move(empty));
        check_variant_empty(a);

        variant_t b(std::move(non_empty1));
        check_variant_value(b, 5);

        variant_t c(std::move(non_empty2));
        check_variant_value(c, debugger_type(42));
        REQUIRE(c.value(variant_type<debugger_type>{}).move_ctor());
    }
    SUBCASE("copy assignment")
    {
        variant_t a;
        a = empty;
        check_variant_empty(a);
        a = non_empty1;
        check_variant_value(a, 5);
        a = non_empty2;
        check_variant_value(a, debugger_type(42));
        REQUIRE(a.value(variant_type<debugger_type>{}).copy_ctor());

        variant_t b(5);
        b = non_empty1;
        check_variant_value(b, 5);
        b = non_empty2;
        check_variant_value(b, debugger_type(42));
        REQUIRE(b.value(variant_type<debugger_type>{}).copy_ctor());
        b = empty;
        check_variant_empty(b);

        variant_t c(debugger_type(42));
        c = non_empty2;
        check_variant_value(c, debugger_type(42));
        REQUIRE(c.value(variant_type<debugger_type>{}).copy_assigned());
        c = non_empty1;
        check_variant_value(c, 5);
        c = empty;
        check_variant_empty(c);
    }
    SUBCASE("move assignment")
    {
        variant_t a;
        a = std::move(empty);
        check_variant_empty(a);
        a = std::move(non_empty1);
        check_variant_value(a, 5);
        a = std::move(non_empty2);
        check_variant_value(a, debugger_type(42));
        REQUIRE(a.value(variant_type<debugger_type>{}).move_ctor());

        variant_t b(5);
        b = std::move(non_empty1);
        check_variant_value(b, 5);
        b = std::move(non_empty2);
        check_variant_value(b, debugger_type(42));
        REQUIRE(b.value(variant_type<debugger_type>{}).move_ctor());
        b = std::move(empty);
        check_variant_empty(b);

        variant_t c(debugger_type(42));
        c = std::move(non_empty2);
        check_variant_value(c, debugger_type(42));
        REQUIRE(c.value(variant_type<debugger_type>{}).move_assigned());
        c = std::move(non_empty1);
        check_variant_value(c, 5);
        c = std::move(empty);
        check_variant_empty(c);
    }
    SUBCASE("swap")
    {
        SUBCASE("empty, empty")
        {
            swap(empty, empty);
            check_variant_empty(empty);

            variant_t empty2(empty);
            swap(empty, empty2);
            check_variant_empty(empty);
            check_variant_empty(empty2);
        }
        SUBCASE("empty, non_empty1")
        {
            swap(empty, non_empty1);
            check_variant_value(empty, 5);
            check_variant_empty(non_empty1);

            swap(empty, non_empty1);
            check_variant_value(non_empty1, 5);
            check_variant_empty(empty);
        }
        SUBCASE("empty, non_empty2")
        {
            swap(empty, non_empty2);
            check_variant_value(empty, debugger_type(42));
            check_variant_empty(non_empty2);
            REQUIRE(empty.value(variant_type<debugger_type>{}).ctor());

            swap(empty, non_empty2);
            check_variant_value(non_empty2, debugger_type(42));
            check_variant_empty(empty);
            REQUIRE(non_empty2.value(variant_type<debugger_type>{}).ctor());
        }
        SUBCASE("non-empty, different types")
        {
            swap(non_empty1, non_empty2);
            check_variant_value(non_empty1, debugger_type(42));
            check_variant_value(non_empty2, 5);
            REQUIRE(non_empty1.value(variant_type<debugger_type>{}).move_ctor());

            swap(non_empty1, non_empty2);
            check_variant_value(non_empty2, debugger_type(42));
            check_variant_value(non_empty1, 5);
            REQUIRE(non_empty2.value(variant_type<debugger_type>{}).move_ctor());
        }
        SUBCASE("non-empty, same types")
        {
            swap(non_empty2, non_empty2);
            check_variant_value(non_empty2, debugger_type(42));
            REQUIRE(non_empty2.value(variant_type<debugger_type>{}).swapped);

            variant_t other(debugger_type(43));
            swap(non_empty2, other);
            check_variant_value(non_empty2, debugger_type(43));
            check_variant_value(other, debugger_type(42));
            REQUIRE(non_empty2.value(variant_type<debugger_type>{}).swapped);
            REQUIRE(other.value(variant_type<debugger_type>{}).swapped);

            swap(non_empty2, other);
            check_variant_value(non_empty2, debugger_type(42));
            check_variant_value(other, debugger_type(43));
            REQUIRE(non_empty2.value(variant_type<debugger_type>{}).swapped);
            REQUIRE(other.value(variant_type<debugger_type>{}).swapped);
        }
    }
    SUBCASE("reset")
    {
        empty.reset();
        check_variant_empty(empty);

        non_empty1.reset();
        check_variant_empty(non_empty1);

        non_empty2.reset();
        check_variant_empty(non_empty2);
    }
    SUBCASE("reset assignment")
    {
        empty = nullvar;
        check_variant_empty(empty);

        non_empty1 = nullvar;
        check_variant_empty(non_empty1);

        non_empty2 = nullvar;
        check_variant_empty(non_empty2);
    }
    SUBCASE("emplace single arg")
    {
        empty.emplace(variant_type<debugger_type>{}, debugger_type(43));
        check_variant_value(empty, debugger_type(43));
        REQUIRE(empty.value(variant_type<debugger_type>{}).move_ctor());

        non_empty1.emplace(variant_type<debugger_type>{}, debugger_type(43));
        check_variant_value(non_empty1, debugger_type(43));
        REQUIRE(non_empty1.value(variant_type<debugger_type>{}).move_ctor());

        non_empty2.emplace(variant_type<debugger_type>{}, debugger_type(43));
        check_variant_value(non_empty1, debugger_type(43));
        REQUIRE(non_empty2.value(variant_type<debugger_type>{}).move_assigned());
    }
    SUBCASE("emplace assignment")
    {
        empty = debugger_type(43);
        check_variant_value(empty, debugger_type(43));
        REQUIRE(empty.value(variant_type<debugger_type>{}).move_ctor());

        non_empty1 = debugger_type(43);
        check_variant_value(non_empty1, debugger_type(43));
        REQUIRE(non_empty1.value(variant_type<debugger_type>{}).move_ctor());

        non_empty2 = debugger_type(43);
        check_variant_value(non_empty1, debugger_type(43));
        REQUIRE(non_empty2.value(variant_type<debugger_type>{}).move_assigned());
    }
    SUBCASE("emplace multiple args")
    {
        empty.emplace(variant_type<debugger_type>{}, 43, 5.0, 'a');
        check_variant_value(empty, debugger_type(43));
        REQUIRE(empty.value(variant_type<debugger_type>{}).ctor());

        non_empty1.emplace(variant_type<debugger_type>{}, 43, 5.0, 'a');
        check_variant_value(non_empty1, debugger_type(43));
        REQUIRE(non_empty1.value(variant_type<debugger_type>{}).ctor());

        non_empty2.emplace(variant_type<debugger_type>{}, 43, 5.0, 'a');
        check_variant_value(non_empty1, debugger_type(43));
        REQUIRE(non_empty2.value(variant_type<debugger_type>{}).ctor());
    }
    SUBCASE("value_or")
    {
        REQUIRE(empty.value_or(variant_type<int>{}, 3) == 3);
        REQUIRE(non_empty1.value_or(variant_type<int>{}, 3) == 5);
        REQUIRE(non_empty2.value_or(variant_type<int>{}, 3) == 3);

        REQUIRE(non_empty2.value_or(variant_type<double>{}, 3.14) == 3.14);
    }
    SUBCASE("map")
    {
        struct functor_t
        {
            bool expect_call = false;

            int operator()(int i, int j)
            {
                REQUIRE(expect_call);
                REQUIRE(i == 5);
                REQUIRE(j == 0);
                return 12;
            }

            int operator()(const debugger_type& dbg, int j)
            {
                REQUIRE(expect_call);
                REQUIRE(dbg.id == 42);
                REQUIRE(j == 0);
                return 42;
            }

            int operator()(double, int) = delete;
        } functor;

        functor.expect_call = false;
        auto a              = empty.map(functor, 0);
        check_variant_empty(a);

        functor.expect_call = true;
        auto b              = non_empty1.map(functor, 0);
        check_variant_value(b, 12);

        functor.expect_call = true;
        auto c              = non_empty2.map(functor, 0);
        check_variant_value(c, 42);

        functor.expect_call = false;
        auto d              = variant_t(3.0).map(functor, 0);
        check_variant_value(d, 3.0);
    }
    SUBCASE("compare null")
    {
        REQUIRE(empty == nullvar);
        REQUIRE_FALSE(empty != nullvar);
        REQUIRE_FALSE(empty < nullvar);
        REQUIRE_FALSE(nullvar < empty);
        REQUIRE(empty <= nullvar);
        REQUIRE(nullvar <= empty);
        REQUIRE_FALSE(empty > nullvar);
        REQUIRE_FALSE(nullvar > empty);
        REQUIRE(empty >= nullvar);
        REQUIRE(nullvar >= empty);

        REQUIRE_FALSE(non_empty1 == nullvar);
        REQUIRE(non_empty1 != nullvar);
        REQUIRE_FALSE(non_empty1 < nullvar);
        REQUIRE(nullvar < non_empty1);
        REQUIRE_FALSE(non_empty1 <= nullvar);
        REQUIRE(nullvar <= non_empty1);
        REQUIRE(non_empty1 > nullvar);
        REQUIRE_FALSE(nullvar > non_empty1);
        REQUIRE(non_empty1 >= nullvar);
        REQUIRE_FALSE(nullvar >= non_empty1);
    }
    SUBCASE("compare value")
    {
        REQUIRE_FALSE(empty == 4);
        REQUIRE(empty != 4);
        REQUIRE(empty < 4);
        REQUIRE_FALSE(4 < empty);
        REQUIRE(empty <= 4);
        REQUIRE_FALSE(4 <= empty);
        REQUIRE_FALSE(empty > 4);
        REQUIRE(4 > empty);
        REQUIRE_FALSE(empty >= 4);
        REQUIRE(4 >= empty);

        REQUIRE_FALSE(non_empty1 == 4);
        REQUIRE(non_empty1 != 4);
        REQUIRE_FALSE(non_empty1 < 4);
        REQUIRE(4 < non_empty1);
        REQUIRE_FALSE(non_empty1 <= 4);
        REQUIRE(4 <= non_empty1);
        REQUIRE(non_empty1 > 4);
        REQUIRE_FALSE(4 > non_empty1);
        REQUIRE(non_empty1 >= 4);
        REQUIRE_FALSE(4 >= non_empty1);

        REQUIRE(non_empty1 == 5);
        REQUIRE_FALSE(non_empty1 != 5);
        REQUIRE_FALSE(non_empty1 < 5);
        REQUIRE_FALSE(5 < non_empty1);
        REQUIRE(non_empty1 <= 5);
        REQUIRE(5 <= non_empty1);
        REQUIRE_FALSE(non_empty1 > 5);
        REQUIRE_FALSE(5 > non_empty1);
        REQUIRE(non_empty1 >= 5);
        REQUIRE(5 >= non_empty1);

        REQUIRE_FALSE(non_empty2 == 4);
        REQUIRE(non_empty2 != 4);
        REQUIRE_FALSE(non_empty2 < 4);
        REQUIRE(4 < non_empty2);
        REQUIRE_FALSE(non_empty2 <= 4);
        REQUIRE(4 <= non_empty2);
        REQUIRE(non_empty2 > 4);
        REQUIRE_FALSE(4 > non_empty2);
        REQUIRE(non_empty2 >= 4);
        REQUIRE_FALSE(4 >= non_empty2);
    }
    SUBCASE("compare variant")
    {
        REQUIRE(empty == variant_t());
        REQUIRE_FALSE(empty != variant_t());
        REQUIRE_FALSE(empty < variant_t());
        REQUIRE_FALSE(variant_t() < empty);
        REQUIRE(empty <= variant_t());
        REQUIRE(variant_t() <= empty);
        REQUIRE_FALSE(empty > variant_t());
        REQUIRE_FALSE(variant_t() > empty);
        REQUIRE(empty >= variant_t());
        REQUIRE(variant_t() >= empty);

        REQUIRE_FALSE(non_empty1 == variant_t());
        REQUIRE(non_empty1 != variant_t());
        REQUIRE_FALSE(non_empty1 < variant_t());
        REQUIRE(variant_t() < non_empty1);
        REQUIRE_FALSE(non_empty1 <= variant_t());
        REQUIRE(variant_t() <= non_empty1);
        REQUIRE(non_empty1 > variant_t());
        REQUIRE_FALSE(variant_t() > non_empty1);
        REQUIRE(non_empty1 >= variant_t());
        REQUIRE_FALSE(variant_t() >= non_empty1);

        REQUIRE_FALSE(empty == variant_t(4));
        REQUIRE(empty != variant_t(4));
        REQUIRE(empty < variant_t(4));
        REQUIRE_FALSE(variant_t(4) < empty);
        REQUIRE(empty <= variant_t(4));
        REQUIRE_FALSE(variant_t(4) <= empty);
        REQUIRE_FALSE(empty > variant_t(4));
        REQUIRE(variant_t(4) > empty);
        REQUIRE_FALSE(empty >= variant_t(4));
        REQUIRE(variant_t(4) >= empty);

        REQUIRE_FALSE(non_empty1 == variant_t(4));
        REQUIRE(non_empty1 != variant_t(4));
        REQUIRE_FALSE(non_empty1 < variant_t(4));
        REQUIRE(variant_t(4) < non_empty1);
        REQUIRE_FALSE(non_empty1 <= variant_t(4));
        REQUIRE(variant_t(4) <= non_empty1);
        REQUIRE(non_empty1 > variant_t(4));
        REQUIRE_FALSE(variant_t(4) > non_empty1);
        REQUIRE(non_empty1 >= variant_t(4));
        REQUIRE_FALSE(variant_t(4) >= non_empty1);

        REQUIRE(non_empty1 == variant_t(5));
        REQUIRE_FALSE(non_empty1 != variant_t(5));
        REQUIRE_FALSE(non_empty1 < variant_t(5));
        REQUIRE_FALSE(variant_t(5) < non_empty1);
        REQUIRE(non_empty1 <= variant_t(5));
        REQUIRE(variant_t(5) <= non_empty1);
        REQUIRE_FALSE(non_empty1 > variant_t(5));
        REQUIRE_FALSE(variant_t(5) > non_empty1);
        REQUIRE(non_empty1 >= variant_t(5));
        REQUIRE(variant_t(5) >= non_empty1);

        REQUIRE_FALSE(non_empty2 == variant_t(4));
        REQUIRE(non_empty2 != variant_t(4));
        REQUIRE_FALSE(non_empty2 < variant_t(4));
        REQUIRE(variant_t(4) < non_empty2);
        REQUIRE_FALSE(non_empty2 <= variant_t(4));
        REQUIRE(variant_t(4) <= non_empty2);
        REQUIRE(non_empty2 > variant_t(4));
        REQUIRE_FALSE(variant_t(4) > non_empty2);
        REQUIRE(non_empty2 >= variant_t(4));
        REQUIRE_FALSE(variant_t(4) >= non_empty2);
    }
    SUBCASE("with")
    {
        struct visitor
        {
            int i;

            void operator()(int val) const
            {
                REQUIRE(i == 1);
                REQUIRE(val == 5);
            }

            void operator()(double f) const
            {
                REQUIRE(i == 2);
                REQUIRE(f == 3.14);
            }
        } v;

        variant_t a;
        v.i = 0;
        with(a, v);

        variant_t b(5);
        v.i = 1;
        with(b, v);

        variant_t c(3.14);
        v.i = 2;
        with(c, v);

        variant_t d(5);
        with(d, [](int& i) { ++i; });
        check_variant_value(d, 6);
    }
}

struct evil_variant_test_type
{
    bool move_throw = false;

    evil_variant_test_type(int) noexcept {}

    evil_variant_test_type(const char*)
    {
        throw "buh!";
    }

    evil_variant_test_type(double)
    {
        move_throw = true;
    }

    evil_variant_test_type(evil_variant_test_type&& other)
    {
        if (other.move_throw)
            throw "haha!";
    }
};

inline bool operator==(const evil_variant_test_type&, const evil_variant_test_type&)
{
    return true;
}

TEST_CASE("fallback_variant")
{
    fallback_variant<int, evil_variant_test_type> var(3);

    SUBCASE("harmless ctor")
    {
        var.emplace(variant_type<evil_variant_test_type>{}, 42);
        check_variant_value(var, evil_variant_test_type(42));
    }
    SUBCASE("move ctor")
    {
        var = evil_variant_test_type(42);
        check_variant_value(var, evil_variant_test_type(42));
    }
    SUBCASE("throwing move")
    {
        evil_variant_test_type test(42);
        test.move_throw = true;
        try
        {
            var = std::move(test);
        }
        catch (...)
        {}
        check_variant_value(var, 0);
    }
}

TEST_CASE("optional_variant")
{
    variant<nullvar_t, int, evil_variant_test_type> var(3);

    SUBCASE("harmless ctor")
    {
        var.emplace(variant_type<evil_variant_test_type>{}, 42);
        check_variant_value(var, evil_variant_test_type(42));
    }
    SUBCASE("move ctor")
    {
        var = evil_variant_test_type(42);
        check_variant_value(var, evil_variant_test_type(42));
    }
    SUBCASE("throwing move")
    {
        evil_variant_test_type test(42);
        test.move_throw = true;
        try
        {
            var = std::move(test);
        }
        catch (...)
        {}
        check_variant_empty(var);
    }
}

TEST_CASE("rarely_empty_variant")
{
    variant<int, evil_variant_test_type> var(3);

    SUBCASE("harmless ctor")
    {
        var.emplace(variant_type<evil_variant_test_type>{}, 42);
        check_variant_value(var, evil_variant_test_type(42));
    }
    SUBCASE("move ctor")
    {
        var = evil_variant_test_type(42);
        check_variant_value(var, evil_variant_test_type(42));
    }
    SUBCASE("throwing ctor")
    {
        try
        {
            var.emplace(variant_type<evil_variant_test_type>{}, "I will throw");
        }
        catch (...)
        {}
        check_variant_value(var, 3);
    }
    SUBCASE("delayed throwing move")
    {
        try
        {
            var.emplace(variant_type<evil_variant_test_type>{}, 3.14);
        }
        catch (...)
        {}
        check_variant_empty(var);
    }
    SUBCASE("throwing move")
    {
        evil_variant_test_type test(42);
        test.move_throw = true;
        try
        {
            var = std::move(test);
        }
        catch (...)
        {}
        check_variant_empty(var);
    }
}
