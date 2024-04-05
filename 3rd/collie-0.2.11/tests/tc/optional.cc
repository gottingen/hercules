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
#include <collie/type_safe/optional.h>
#include <collie/type_safe/visitor.h>

#include <collie/testing/doctest.h>

#include "debugger_type.hpp"

using collie::ts::optional;
using collie::ts::optional_ref;
using collie::ts::make_optional;
using collie::ts::nullopt;

TEST_CASE("optional")
{
    SUBCASE("constructor - empty")
    {
        optional<int> a;
        REQUIRE_FALSE(a.has_value());

        optional<int> b(nullopt);
        REQUIRE_FALSE(b.has_value());
    }
    SUBCASE("constructor - value")
    {
        optional<debugger_type> a(debugger_type(3));
        REQUIRE(a.has_value());
        REQUIRE(a.value().id == 3);
        REQUIRE(a.value().move_ctor());

        debugger_type           dbg(2);
        optional<debugger_type> b(dbg);
        REQUIRE(b.has_value());
        REQUIRE(b.value().id == 2);
        REQUIRE(b.value().copy_ctor());

        optional<debugger_type> c(std::move(dbg));
        REQUIRE(c.has_value());
        REQUIRE(c.value().id == 2);
        REQUIRE(c.value().move_ctor());

        optional<debugger_type> d(0);
        REQUIRE(d.has_value());
        REQUIRE(d.value().id == 0);
        REQUIRE(d.value().ctor());
    }
    SUBCASE("constructor - move/copy")
    {
        optional<debugger_type> org_empty;
        optional<debugger_type> org_value(debugger_type(0));

        optional<debugger_type> a(org_empty);
        REQUIRE_FALSE(a.has_value());

        optional<debugger_type> b(std::move(org_empty));
        REQUIRE_FALSE(b.has_value());

        optional<debugger_type> c(org_value);
        REQUIRE(c.has_value());
        REQUIRE(c.value().id == 0);
        REQUIRE(c.value().copy_ctor());

        optional<debugger_type> d(std::move(org_value));
        REQUIRE(d.has_value());
        REQUIRE(d.value().id == 0);
        REQUIRE(d.value().move_ctor());
    }
    SUBCASE("assignment - nullopt_t")
    {
        optional<debugger_type> a;
        a = nullopt;
        REQUIRE_FALSE(a.has_value());

        optional<debugger_type> b(4);
        b = nullopt;
        REQUIRE_FALSE(b.has_value());
    }
    SUBCASE("assignment - value")
    {
        optional<debugger_type> a;
        a = debugger_type(0);
        REQUIRE(a.has_value());
        REQUIRE(a.value().id == 0);
        REQUIRE(a.value().move_ctor());
        REQUIRE(a.value().not_assigned());

        debugger_type           dbg(0);
        optional<debugger_type> b;
        b = dbg;
        REQUIRE(b.has_value());
        REQUIRE(b.value().id == 0);
        REQUIRE(b.value().copy_ctor());
        REQUIRE(b.value().not_assigned());

        optional<debugger_type> c;
        c = 0;
        REQUIRE(c.has_value());
        REQUIRE(c.value().id == 0);
        REQUIRE(c.value().ctor());
        REQUIRE(c.value().not_assigned());

        optional<debugger_type> d(0);
        d = debugger_type(1);
        REQUIRE(d.has_value());
        REQUIRE(d.value().id == 1);
        REQUIRE(d.value().ctor());
        REQUIRE(d.value().move_assigned());

        dbg.id = 1;
        optional<debugger_type> e(0);
        e = dbg;
        REQUIRE(e.has_value());
        REQUIRE(e.value().id == 1);
        REQUIRE(e.value().ctor());
        REQUIRE(e.value().copy_assigned());

        optional<debugger_type> f(0);
        f = 1; // assignment would use implicit conversion, so it destroys & recreates
        REQUIRE(f.has_value());
        REQUIRE(f.value().id == 1);
        REQUIRE(f.value().ctor());
        REQUIRE(f.value().not_assigned());
    }
    SUBCASE("assignment - move/copy")
    {
        optional<debugger_type> new_empty;
        optional<debugger_type> new_value(debugger_type(1));

        optional<debugger_type> a;
        a = new_empty;
        REQUIRE_FALSE(a.has_value());

        optional<debugger_type> b;
        b = new_value;
        REQUIRE(b.has_value());
        REQUIRE(b.value().id == 1);
        REQUIRE(b.value().copy_ctor());
        REQUIRE(b.value().not_assigned());

        optional<debugger_type> c;
        c = std::move(new_value);
        REQUIRE(c.has_value());
        REQUIRE(c.value().id == 1);
        REQUIRE(c.value().not_assigned());

        optional<debugger_type> d(0);
        d = new_empty;
        REQUIRE_FALSE(d.has_value());

        optional<debugger_type> e(0);
        e = new_value;
        REQUIRE(e.has_value());
        REQUIRE(e.value().id == 1);
        REQUIRE(e.value().ctor());
        REQUIRE(e.value().copy_assigned());

        optional<debugger_type> f(0);
        f = std::move(new_value);
        REQUIRE(f.has_value());
        REQUIRE(f.value().id == 1);
        REQUIRE(f.value().ctor());
    }
    SUBCASE("swap")
    {
        optional<debugger_type> empty1, empty2;
        optional<debugger_type> a(0);
        optional<debugger_type> b(1);

        SUBCASE("empty, empty")
        {
            swap(empty1, empty2);
            REQUIRE_FALSE(empty1.has_value());
            REQUIRE_FALSE(empty2.has_value());
        }
        SUBCASE("value, value")
        {
            swap(a, b);
            REQUIRE(a.has_value());
            REQUIRE(a.value().id == 1);
            REQUIRE(a.value().swapped);
            REQUIRE(b.has_value());
            REQUIRE(b.value().id == 0);
            REQUIRE(b.value().swapped);
        }
        SUBCASE("empty, value")
        {
            swap(empty1, a);
            REQUIRE_FALSE(a.has_value());
            REQUIRE(empty1.has_value());
            REQUIRE(empty1.value().id == 0);
            REQUIRE(empty1.value().not_assigned());
            REQUIRE_FALSE(empty1.value().swapped);
        }
        SUBCASE("value, empty")
        {
            swap(a, empty1);
            REQUIRE_FALSE(a.has_value());
            REQUIRE(empty1.has_value());
            REQUIRE(empty1.value().id == 0);
            REQUIRE(empty1.value().move_ctor());
            REQUIRE(empty1.value().not_assigned());
            REQUIRE_FALSE(empty1.value().swapped);
        }
    }
    SUBCASE("reset")
    {
        optional<debugger_type> a;
        a.reset();
        REQUIRE_FALSE(a.has_value());

        optional<debugger_type> b(0);
        b.reset();
        REQUIRE_FALSE(b.has_value());
    }
    SUBCASE("emplace")
    {
        debugger_type dbg(1);

        optional<debugger_type> a;
        a.emplace(1);
        REQUIRE(a.has_value());
        REQUIRE(a.value().id == 1);
        REQUIRE(a.value().ctor());
        REQUIRE(a.value().not_assigned());

        optional<debugger_type> b;
        b.emplace(dbg);
        REQUIRE(b.has_value());
        REQUIRE(b.value().id == 1);
        REQUIRE(b.value().copy_ctor());
        REQUIRE(b.value().not_assigned());

        optional<debugger_type> c;
        c.emplace(std::move(dbg));
        REQUIRE(c.has_value());
        REQUIRE(c.value().id == 1);
        REQUIRE(c.value().not_assigned());

        optional<debugger_type> d(0);
        d.emplace(1);
        REQUIRE(d.has_value());
        REQUIRE(d.value().id == 1);
        REQUIRE(d.value().ctor());
        REQUIRE(d.value().not_assigned());

        optional<debugger_type> e(0);
        e.emplace(dbg);
        REQUIRE(e.has_value());
        REQUIRE(e.value().id == 1);
        REQUIRE(e.value().ctor());
        REQUIRE(e.value().copy_assigned());

        optional<debugger_type> f(0);
        f.emplace(std::move(dbg));
        REQUIRE(f.has_value());
        REQUIRE(f.value().id == 1);
        REQUIRE(f.value().ctor());
        REQUIRE(f.value().move_assigned());
    }
    SUBCASE("operator bool")
    {
        optional<int> a;
        REQUIRE_FALSE(static_cast<bool>(a));

        optional<int> b(0);
        REQUIRE(static_cast<bool>(b));
    }
    SUBCASE("value")
    {
        // only test the return types
        optional<debugger_type> a(0);
        static_assert(std::is_same<decltype(a.value()), debugger_type&>::value, "");
        static_assert(std::is_same<decltype(std::move(a.value())), debugger_type&&>::value, "");

        const optional<debugger_type> b(0);
        static_assert(std::is_same<decltype(b.value()), const debugger_type&>::value, "");
        static_assert(std::is_same<decltype(std::move(b.value())), const debugger_type&&>::value,
                      "");
    }
    SUBCASE("value_or")
    {
        optional<debugger_type> a;
        auto                    a_res = a.value_or(1);
        REQUIRE(a_res.id == 1);

        optional<debugger_type> b(0);
        auto                    b_res = b.value_or(1);
        REQUIRE(b_res.id == 0);
    }
    SUBCASE("map")
    {
        auto func = [](int i) { return "abc"[i]; };

        optional<int>  a;
        optional<char> a_res = a.map(func);
        REQUIRE_FALSE(a_res.has_value());

        optional<int>  b(0);
        optional<char> b_res = b.map(func);
        REQUIRE(b_res.has_value());
        REQUIRE(b_res.value() == 'a');

        struct foo
        {
            int var = 42;

            int func(int i)
            {
                return 2 * i;
            }

            void func2(int i)
            {
                REQUIRE(i == 42);
            }
        };

        optional<foo> c(foo{});

        optional<int> c_res = c.map(&foo::func, 2);
        REQUIRE(c_res.has_value());
        REQUIRE(c_res.value() == 4);

        c.map(&foo::func2, 42);

        optional_ref<int> c_res2 = c.map(&foo::var);
        REQUIRE(c_res2.has_value());
        REQUIRE(c_res2.value() == 42);

#if TYPE_SAFE_USE_RETURN_TYPE_DEDUCTION
        // just compiler check, see https://github.com/foonathan/type_safe/issues/60
        struct bar
        {
            void non_const() {}
        };

        optional<bar> f = bar{};
        f.map([](auto&& b) {
            b.non_const();
            return 42;
        });
#endif
    }
    SUBCASE("with")
    {
        optional<int> a;
        with(a, [](int) { REQUIRE(false); });

        a = 0;
        with(a, [](int& i) {
            REQUIRE(i == 0);
            i = 1;
        });
        REQUIRE(a.has_value());
        REQUIRE(a.value() == 1);
    }
    SUBCASE("comparison")
    {
        optional<int> a;
        optional<int> b(1);
        optional<int> c(2);

        // ==
        REQUIRE(b == b);
        REQUIRE(!(b == c));
        REQUIRE(!(b == a));

        REQUIRE(a == nullopt);
        REQUIRE(nullopt == a);
        REQUIRE(!(b == nullopt));
        REQUIRE(!(nullopt == b));

        REQUIRE(b == 1);
        REQUIRE(!(a == 1));
        REQUIRE(!(1 == a));
        REQUIRE(!(c == 1));
        REQUIRE(!(1 == c));

        // !=
        REQUIRE(a != b);
        REQUIRE(b != c);
        REQUIRE(!(a != a));

        REQUIRE(b != nullopt);
        REQUIRE(nullopt != b);
        REQUIRE(!(a != nullopt));
        REQUIRE(!(nullopt != a));

        REQUIRE(b != 2);
        REQUIRE(2 != b);
        REQUIRE(a != 2);
        REQUIRE(2 != a);
        REQUIRE(!(c != 2));
        REQUIRE(!(2 != c));

        // <
        REQUIRE(a < b);
        REQUIRE(b < c);
        REQUIRE(!(c < b));
        REQUIRE(!(b < a));

        REQUIRE(!(a < nullopt));
        REQUIRE(!(nullopt < a));
        REQUIRE(!(b < nullopt));
        REQUIRE(nullopt < b);

        REQUIRE(a < 2);
        REQUIRE(!(2 < a));
        REQUIRE(!(c < 2));
        REQUIRE(!(2 < c));

        // <=
        REQUIRE(a <= b);
        REQUIRE(b <= c);
        REQUIRE(b <= b);
        REQUIRE(!(c <= b));

        REQUIRE(a <= nullopt);
        REQUIRE(nullopt <= a);
        REQUIRE(!(b <= nullopt));
        REQUIRE(nullopt <= b);

        REQUIRE(a <= 2);
        REQUIRE(!(2 <= a));
        REQUIRE(b <= 2);
        REQUIRE(!(2 <= b));
        REQUIRE(c <= 2);
        REQUIRE(2 <= c);

        // >
        REQUIRE(c > b);
        REQUIRE(b > a);
        REQUIRE(!(a > b));

        REQUIRE(b > nullopt);
        REQUIRE(!(nullopt > b));
        REQUIRE(!(a > nullopt));
        REQUIRE(!(nullopt > b));

        REQUIRE(c > 1);
        REQUIRE(!(1 > c));
        REQUIRE(!(b > 1));
        REQUIRE(!(1 > b));
        REQUIRE(!(a > 1));
        REQUIRE(1 > a);

        // >=
        REQUIRE(c >= b);
        REQUIRE(b >= a);
        REQUIRE(a >= a);
        REQUIRE(!(a >= b));

        REQUIRE(a >= nullopt);
        REQUIRE(nullopt >= a);
        REQUIRE(b >= nullopt);
        REQUIRE(!(nullopt >= b));

        REQUIRE(b >= 1);
        REQUIRE(1 >= b);
        REQUIRE(c >= 1);
        REQUIRE(!(1 >= c));
        REQUIRE(!(a >= 1));
        REQUIRE(1 >= a);
    }
    SUBCASE("make_optional")
    {
        optional<int> a = make_optional(5);
        REQUIRE(a.has_value());
        REQUIRE(a.value() == 5);

        optional<std::string> b = make_optional<std::string>(1u, 'a');
        REQUIRE(b.has_value());
        REQUIRE(b.value() == "a");
    }
}
