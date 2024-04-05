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
#include <collie/type_safe/flag_set.h>

#include <collie/testing/doctest.h>

// no using namespace to test operator namespace

enum class test_flags
{
    a,
    b,
    c
};

namespace collie::ts
{
template <>
struct flag_set_traits<test_flags> : std::true_type
{
    static constexpr std::size_t size()
    {
        return 3;
    }
};
} // namespace collie::ts

void check_set(const collie::ts::flag_set<test_flags>& set, bool a, bool b, bool c)
{
    REQUIRE(set.is_set(test_flags::a) == a);
    REQUIRE(set.as_flag(test_flags::a) == a);
    if (a)
        REQUIRE((set & test_flags::a));

    REQUIRE(set.is_set(test_flags::b) == b);
    REQUIRE(set.as_flag(test_flags::b) == b);
    if (b)
        REQUIRE((set & test_flags::b));

    REQUIRE(set.is_set(test_flags::c) == c);
    REQUIRE(set.as_flag(test_flags::c) == c);
    if (c)
        REQUIRE((set & test_flags::c));

    if (a || b || c)
    {
        REQUIRE(set.any());
        REQUIRE_FALSE(set.none());
        REQUIRE(set != collie::ts::noflag);
        REQUIRE(collie::ts::noflag != set);
    }
    else
    {
        REQUIRE_FALSE(set.any());
        REQUIRE(set.none());
        REQUIRE(set == collie::ts::noflag);
        REQUIRE(collie::ts::noflag == set);
    }

    auto number = (a ? 4 : 0) + (b ? 2 : 0) + (c ? 1 : 0);
    switch (number)
    {
    case 0:
        REQUIRE(set == collie::ts::flag_set<test_flags>());
        break;
    case 1:
        REQUIRE(set == test_flags::c);
        REQUIRE(set == collie::ts::combo(~test_flags::a & ~test_flags::b));
        REQUIRE((set & test_flags::c));
        break;
    case 2:
        REQUIRE(set == test_flags::b);
        REQUIRE(set == collie::ts::combo(~test_flags::a & ~test_flags::c));
        REQUIRE((set & test_flags::b));
        break;
    case 3:
        REQUIRE(set == (test_flags::b | test_flags::c));
        REQUIRE(set == collie::ts::combo(~test_flags::a));
        REQUIRE((set & (test_flags::b | test_flags::c)));
        break;
    case 4:
        REQUIRE(set == test_flags::a);
        REQUIRE(set == collie::ts::combo(~test_flags::b & ~test_flags::c));
        REQUIRE((set & test_flags::a));
        break;
    case 5:
        REQUIRE(set == (test_flags::a | test_flags::c));
        REQUIRE(set == collie::ts::combo(~test_flags::b));
        REQUIRE((set & (test_flags::a | test_flags::c)));
        break;
    case 6:
        REQUIRE(set == (test_flags::a | test_flags::b));
        REQUIRE(set == collie::ts::combo(~test_flags::c));
        REQUIRE((set & (test_flags::a | test_flags::b)));
        break;
    case 7:
        REQUIRE(set.all());
        REQUIRE(set == (test_flags::a | test_flags::b | test_flags::c));
        REQUIRE((set & (test_flags::a | test_flags::b | test_flags::c)));
        break;

    default:
        REQUIRE(false);
        break;
    }
}

TEST_CASE("flag_set_traits")
{
    using namespace collie::ts;

    enum class a
    {
        e1,
        e2,
        e3
    };
    static_assert(!flag_set_traits<a>::value, "a is not a flag set enum");

    enum class b
    {
        e1,
        e2,
        e3,
        _flag_set_size,
    };
    static_assert(flag_set_traits<b>::value, "b is a flag set enum");
    static_assert(flag_set_traits<b>::size() == 3u, "size of b is 3");
}

TEST_CASE("flag_set")
{
    using namespace collie::ts;

    using set = flag_set<test_flags>;

    set s;
    check_set(s, false, false, false);

    SUBCASE("constructor/assignment")
    {
        set a;
        check_set(a, false, false, false);

        set b(test_flags::a);
        check_set(b, true, false, false);

        a = test_flags::b;
        check_set(a, false, true, false);

        b = test_flags::c;
        check_set(b, false, false, true);
    }
    SUBCASE("set")
    {
        s.set(test_flags::a);
        check_set(s, true, false, false);

        s.set(test_flags::b, true);
        check_set(s, true, true, false);

        s.set(test_flags::a, false);
        check_set(s, false, true, false);

        s.set(test_flags::a, flag(true));
        check_set(s, true, true, false);
    }
    SUBCASE("reset")
    {
        s.set(test_flags::a);
        check_set(s, true, false, false);

        s.reset(test_flags::b);
        check_set(s, true, false, false);

        s.reset(test_flags::a);
        check_set(s, false, false, false);
    }
    SUBCASE("toggle")
    {
        s.toggle(test_flags::a);
        check_set(s, true, false, false);

        s.toggle(test_flags::b);
        check_set(s, true, true, false);

        s.toggle(test_flags::a);
        check_set(s, false, true, false);
    }
    SUBCASE("set_all/reset_all/toggle_all")
    {
        s.set_all();
        check_set(s, true, true, true);

        s.reset_all();
        check_set(s, false, false, false);

        s.set(test_flags::c);
        check_set(s, false, false, true);

        s.toggle_all();
        check_set(s, true, true, false);
    }
    SUBCASE("binary op")
    {
        s |= test_flags::a;
        check_set(s, true, false, false);

        s |= test_flags::b | test_flags::c;
        check_set(s, true, true, true);

        s &= ~test_flags::c;
        check_set(s, true, true, false);

        s = ~s;
        check_set(s, false, false, true);

        s ^= test_flags::a | test_flags::c;
        check_set(s, true, false, false);
    }
}
