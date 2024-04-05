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
#include <collie/type_safe/strong_typedef.h>

#include <collie/testing/doctest.h>

#include <sstream>

using namespace collie::ts;

#define CREATE_IS_OPERATOR_CALLABLE_WITH_ARGS_CHECKER(Op, CheckerName)                             \
    template <typename Arg1, typename Arg2, typename = void>                                       \
    struct CheckerName : std::false_type                                                           \
    {};                                                                                            \
    template <typename Arg1, typename Arg2>                                                        \
    struct CheckerName<Arg1, Arg2,                                                                 \
                       detail::void_t<decltype(static_cast<Arg1>(                                  \
                           std::declval<Arg1>()) Op static_cast<Arg2>(std::declval<Arg2>()))>>     \
    : std::true_type                                                                               \
    {};

CREATE_IS_OPERATOR_CALLABLE_WITH_ARGS_CHECKER(+, is_operator_plus_callable_with)
CREATE_IS_OPERATOR_CALLABLE_WITH_ARGS_CHECKER(-, is_operator_minus_callable_with)
CREATE_IS_OPERATOR_CALLABLE_WITH_ARGS_CHECKER(/, is_division_callable_with)

TEST_CASE("strong_typedef")
{
    // only check compilation here
    SUBCASE("general")
    {
        struct type : strong_typedef<type, int>, strong_typedef_op::equality_comparison<type>
        {
            using strong_typedef::strong_typedef;
        };

        // type + type
        type t1, t2;
        REQUIRE(t1 == t2);
        REQUIRE(t1 == std::move(t2));
        REQUIRE(std::move(t1) == t2);
        REQUIRE(std::move(t1) == std::move(t2));

        // type + convert_a
        struct convert_a : type
        {
            using type::type;
        };
        convert_a a;
        REQUIRE(t1 == a);
        REQUIRE(t1 == std::move(a));
        REQUIRE(std::move(t1) == a);
        REQUIRE(std::move(t1) == std::move(a));

        REQUIRE(a == a);
        REQUIRE(std::move(a) == a);
        REQUIRE(a == std::move(a));
        REQUIRE(std::move(a) == std::move(a));

        // type + convert_b
        struct convert_b
        {
            operator type() const
            {
                return type(0);
            }
        };
        convert_b b;
        REQUIRE(t1 == b);
        REQUIRE(t1 == std::move(b));
        REQUIRE(std::move(t1) == b);
        REQUIRE(std::move(t1) == std::move(b));
    }
    SUBCASE("general mixed")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::mixed_equality_comparison<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        int i = 0;

        // type + int
        type t1;
        REQUIRE(t1 == i);
        REQUIRE(t1 == std::move(i));
        REQUIRE(std::move(t1) == i);
        REQUIRE(std::move(t1) == std::move(i));

        // type + convert
        struct convert
        {
            operator int() const
            {
                return 0;
            }
        };
        convert a;
        REQUIRE(t1 == a);
        REQUIRE(t1 == std::move(a));
        REQUIRE(std::move(t1) == a);
        REQUIRE(std::move(t1) == std::move(a));
    }
    SUBCASE("general mixed + non mixed")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::equality_comparison<type>,
                      strong_typedef_op::mixed_equality_comparison<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        // type + type
        type t1, t2;
        REQUIRE(t1 == t2);
        REQUIRE(t1 == std::move(t2));
        REQUIRE(std::move(t1) == t2);
        REQUIRE(std::move(t1) == std::move(t2));

        // type + convert_a
        struct convert_a : type
        {
            using type::type;
        };
        convert_a a;
        REQUIRE(t1 == a);
        REQUIRE(t1 == std::move(a));
        REQUIRE(std::move(t1) == a);
        REQUIRE(std::move(t1) == std::move(a));

        REQUIRE(a == a);
        REQUIRE(std::move(a) == a);
        REQUIRE(a == std::move(a));
        REQUIRE(std::move(a) == std::move(a));

        // type + convert_b
        struct convert_b
        {
            operator type() const
            {
                return type(0);
            }
        };
        convert_b b;
        REQUIRE(t1 == b);
        REQUIRE(t1 == std::move(b));
        REQUIRE(std::move(t1) == b);
        REQUIRE(std::move(t1) == std::move(b));

        // type + int
        int i = 0;
        REQUIRE(t1 == i);
        REQUIRE(t1 == std::move(i));
        REQUIRE(std::move(t1) == i);
        REQUIRE(std::move(t1) == std::move(i));

        // type + convert
        struct convert_c
        {
            operator int() const
            {
                return 0;
            }
        };
        convert_c c;
        REQUIRE(t1 == c);
        REQUIRE(t1 == std::move(c));
        REQUIRE(std::move(t1) == c);
        REQUIRE(std::move(t1) == std::move(c));
    }
    SUBCASE("equality_comparison")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::equality_comparison<type>,
                      strong_typedef_op::mixed_equality_comparison<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0);
        type b(1);

        REQUIRE(a == a);
        REQUIRE(a == 0);
        REQUIRE(0 == a);
        REQUIRE(!(a == b));

        REQUIRE(a != b);
        REQUIRE(a != 1);
        REQUIRE(1 != a);
        REQUIRE(!(a != a));
    }
    SUBCASE("relational_comparison")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::relational_comparison<type>,
                      strong_typedef_op::mixed_relational_comparison<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0);
        type b(1);

        REQUIRE(a < b);
        REQUIRE(0 < b);
        REQUIRE(a < 1);
        REQUIRE(!(b < a));
        REQUIRE(a <= b);
        REQUIRE(a <= 1);
        REQUIRE(1 <= b);
        REQUIRE(a <= a);
        REQUIRE(b > a);
        REQUIRE(1 > a);
        REQUIRE(b > 0);
        REQUIRE(!(a > b));
        REQUIRE(b >= a);
        REQUIRE(b >= 0);
        REQUIRE(1 >= a);
        REQUIRE(b >= b);
    }
    SUBCASE("addition")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::addition<type>,
                      strong_typedef_op::mixed_addition<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0);
        a += type(1);
        a = a + type(1);
        a = type(1) + a;
        REQUIRE(static_cast<int>(a) == 3);

        type b(0);
        b += 1;
        b = b + 1;
        b = 1 + b;
        REQUIRE(static_cast<int>(b) == 3);
    }
    SUBCASE("addition with other strong_typedef")
    {
        struct type_a : strong_typedef<type_a, int>
        {
            using strong_typedef::strong_typedef;
        };
        struct type_b : strong_typedef<type_b, int>,
                        strong_typedef_op::mixed_addition<type_b, type_a>
        {
            using strong_typedef::strong_typedef;
        };
        type_a a(3);
        type_b b(1);
        b += a;    // 4
        b = b + a; // 7
        b = a + b; // 10
        REQUIRE(static_cast<int>(b) == 10);

        struct type_c : strong_typedef<type_b, int>
        {};

        static_assert(is_operator_plus_callable_with<type_b, type_a>::value,
                      "type_b supports addition with type_a");
        static_assert(is_operator_plus_callable_with<type_a, type_b>::value,
                      "type_b supports commutative addition with type_a");
        static_assert(!is_operator_plus_callable_with<type_b, int>::value,
                      "type_b support addition only with type_a, not with int");
        static_assert(!is_operator_plus_callable_with<type_b, type_c>::value,
                      "type_b support addition only with type_a, not with other strong_typedefs");
    }
    SUBCASE("subtraction")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::subtraction<type>,
                      strong_typedef_op::mixed_subtraction<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0);
        a -= type(1);    // -1
        a = a - type(1); // -2
        a = type(1) - a; // 3
        REQUIRE(static_cast<int>(a) == 3);

        type b(0);
        b -= 1;
        b = b - 1;
        b = 1 - b;
        REQUIRE(static_cast<int>(b) == 3);
    }
    SUBCASE("subtraction noncommutative")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::subtraction<type>,
                      strong_typedef_op::mixed_subtraction_noncommutative<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0);
        a -= type(1);    // -1
        a = a - type(1); // -2
        a = type(1) - a; // 3
        REQUIRE(static_cast<int>(a) == 3);

        type b(0);
        b -= 1;
        b = b - 1;
        REQUIRE(static_cast<int>(b) == -2);
        static_assert(is_operator_minus_callable_with<type, int>::value, "");
        static_assert(!is_operator_minus_callable_with<int, type>::value, "type is noncommutative");
    }
    SUBCASE("multiplication")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::multiplication<type>,
                      strong_typedef_op::mixed_multiplication<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(1);
        a *= type(2);
        a = a * type(2);
        a = type(2) * a;
        REQUIRE(static_cast<int>(a) == 8);

        type b(1);
        b *= 2;
        b = b * 2;
        b = 2 * b;
        REQUIRE(static_cast<int>(b) == 8);
    }
    SUBCASE("division")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::division<type>,
                      strong_typedef_op::mixed_division<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(8);
        a /= type(2);
        a = a / type(2);
        a = type(2) / a;
        REQUIRE(static_cast<int>(a) == 1);

        type b(8);
        b /= 2;
        b = b / 2;
        b = 2 / b;
        REQUIRE(static_cast<int>(b) == 1);
    }
    SUBCASE("division noncommutative")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::division<type>,
                      strong_typedef_op::mixed_division_noncommutative<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(8);
        a /= type(2);
        a = a / type(2);
        a = type(2) / a;
        REQUIRE(static_cast<int>(a) == 1);

        type b(8);
        b /= 2;
        b = b / 2;
        REQUIRE(static_cast<int>(b) == 2);
        static_assert(is_division_callable_with<type, int>::value, "");
        static_assert(!is_division_callable_with<int, type>::value,
                      "division of type and int is noncommutative");
    }
    SUBCASE("modulo")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::modulo<type>,
                      strong_typedef_op::mixed_modulo<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(11);
        a %= type(6);    // 5
        a = a % type(2); // 1
        a = type(2) % a; // 0
        REQUIRE(static_cast<int>(a) == 0);

        type b(11);
        b %= 6;
        b = b % 2;
        b = 2 % b;
        REQUIRE(static_cast<int>(b) == 0);
    }
    SUBCASE("increment")
    {
        struct type : strong_typedef<type, int>, strong_typedef_op::increment<type>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0);
        REQUIRE(static_cast<int>(++a) == 1);
        REQUIRE(static_cast<int>(a++) == 1);
        REQUIRE(static_cast<int>(a) == 2);
    }
    SUBCASE("decrement")
    {
        struct type : strong_typedef<type, int>, strong_typedef_op::decrement<type>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0);
        REQUIRE(static_cast<int>(--a) == -1);
        REQUIRE(static_cast<int>(a--) == -1);
        REQUIRE(static_cast<int>(a) == -2);
    }
    SUBCASE("unary")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::unary_minus<type>,
                      strong_typedef_op::unary_plus<type>
        {
            using strong_typedef::strong_typedef;
        };

        type a(2);
        REQUIRE(static_cast<int>(+a) == 2);
        REQUIRE(static_cast<int>(-a) == -2);
    }
    SUBCASE("complement")
    {
        struct type : strong_typedef<type, unsigned>, strong_typedef_op::complement<type>
        {
            using strong_typedef::strong_typedef;
        };

        type a(1u);
        REQUIRE(static_cast<unsigned>(~a) == ~1u);

        type b(42u);
        REQUIRE(static_cast<unsigned>(~b) == ~42u);
    }
    SUBCASE("bitwise")
    {
        struct type : strong_typedef<type, unsigned>,
                      strong_typedef_op::bitwise_or<type>,
                      strong_typedef_op::bitwise_xor<type>,
                      strong_typedef_op::bitwise_and<type>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0u);

        REQUIRE(static_cast<unsigned>(a | type(3u)) == 3u);
        REQUIRE(static_cast<unsigned>(type(3u) | a) == 3u);
        a |= type(3u);
        REQUIRE(static_cast<unsigned>(a) == 3u);

        REQUIRE(static_cast<unsigned>(a & type(2u)) == 2u);
        REQUIRE(static_cast<unsigned>(type(2u) & a) == 2u);
        a &= type(2u);
        REQUIRE(static_cast<unsigned>(a) == 2u);

        REQUIRE(static_cast<unsigned>(a ^ type(3u)) == 1u);
        REQUIRE(static_cast<unsigned>(type(3u) ^ a) == 1u);
        a ^= type(3u);
        REQUIRE(static_cast<unsigned>(a) == 1u);
    }
    SUBCASE("bitshift")
    {
        struct type : strong_typedef<type, unsigned>, strong_typedef_op::bitshift<type, unsigned>
        {
            using strong_typedef::strong_typedef;
        };

        type a(1u);

        REQUIRE(static_cast<unsigned>(a << 1u) == 2u);
        a <<= 1u;
        REQUIRE(static_cast<unsigned>(a) == 2u);

        REQUIRE(static_cast<unsigned>(a >> 1u) == 1u);
        a >>= 1u;
        REQUIRE(static_cast<unsigned>(a) == 1u);
    }
    SUBCASE("dereference")
    {
        struct test
        {
            int a;
        } t{0};

        struct type : strong_typedef<type, test*>, strong_typedef_op::dereference<type, test>
        {
            using strong_typedef::strong_typedef;
        };

        type a(&t);
        REQUIRE((*a).a == 0);
        REQUIRE(a->a == 0);
    }
    SUBCASE("array subscript")
    {
        int arr[] = {0, 1, 2};

        struct type : strong_typedef<type, int*>, strong_typedef_op::array_subscript<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(arr);
        REQUIRE(a[0] == 0);
        REQUIRE(a[1] == 1);
        REQUIRE(a[2] == 2);
    }
    SUBCASE("iterator")
    {
        int arr[] = {0, 1, 2};

        struct type : strong_typedef<type, int*>,
                      strong_typedef_op::random_access_iterator<type, int>
        {
            using strong_typedef::strong_typedef;
        };

        type a(arr);
        a += 1;
        REQUIRE(a == type(&arr[1]));

        a -= 1;
        REQUIRE(a == type(&arr[0]));

        a = a + 1;
        a = 1 + a;
        REQUIRE(a == type(&arr[2]));

        a = a - 1;
        REQUIRE(a == type(&arr[1]));

        REQUIRE(a - type(&arr[0]) == 1);
    }
    SUBCASE("i/o")
    {
        struct type : strong_typedef<type, int>,
                      strong_typedef_op::input_operator<type>,
                      strong_typedef_op::output_operator<type>
        {
            using strong_typedef::strong_typedef;
        };

        std::ostringstream out;
        std::istringstream in("1");

        type a(0);
        out << a;
        REQUIRE(out.str() == "0");

        in >> a;
        REQUIRE(static_cast<int>(a) == 1);
    }
    SUBCASE("explicit bool")
    {
        struct type : strong_typedef<type, int>, strong_typedef_op::explicit_bool<type>
        {
            using strong_typedef::strong_typedef;
        };

        type a(0);
        REQUIRE(!a);
        type b(1);
        REQUIRE(b);
    }
    SUBCASE("explicit bool nonconstexpr")
    {
        struct foo
        {
            bool flag;

            explicit operator bool() const
            {
                return flag;
            }
        };

        struct type : strong_typedef<type, foo>, strong_typedef_op::explicit_bool<type>
        {
            using strong_typedef::strong_typedef;
        };

        type a(foo{false});
        REQUIRE(!a);
        type b(foo{true});
        REQUIRE(b);
    }
    SUBCASE("is_strong_typedef")
    {
        struct strong : strong_typedef<strong, int>
        {
            using strong_typedef::strong_typedef;
        };

        REQUIRE(is_strong_typedef<strong>::value);
        REQUIRE(!is_strong_typedef<int>::value);
    }
}
