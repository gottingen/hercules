// Copyright 2018 The Turbo Authors.
//
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

#include "turbo/meta/compare.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

#include "turbo/base/casts.h"

namespace turbo {
    namespace {

        // This is necessary to avoid a bunch of lint warnings suggesting that we use
        // CHECK_EQ/etc., which doesn't work in this case because they convert the `0`
        // to an int, which can't be converted to the unspecified zero type.
        bool Identity(bool b) { return b; }

        TEST_CASE("Compare, WeakEquality") {
            CHECK(Identity(weak_equality::equivalent == 0));
            CHECK(Identity(0 == weak_equality::equivalent));
            CHECK(Identity(weak_equality::nonequivalent != 0));
            CHECK(Identity(0 != weak_equality::nonequivalent));
            const weak_equality values[] = {weak_equality::equivalent,
                                            weak_equality::nonequivalent};
            for (const auto &lhs: values) {
                for (const auto &rhs: values) {
                    const bool are_equal = &lhs == &rhs;
                    CHECK_EQ(lhs == rhs, are_equal);
                    CHECK_EQ(lhs != rhs, !are_equal);
                }
            }
        }

        TEST_CASE("Compare, StrongEquality") {
            CHECK(Identity(strong_equality::equal == 0));
            CHECK(Identity(0 == strong_equality::equal));
            CHECK(Identity(strong_equality::nonequal != 0));
            CHECK(Identity(0 != strong_equality::nonequal));
            CHECK(Identity(strong_equality::equivalent == 0));
            CHECK(Identity(0 == strong_equality::equivalent));
            CHECK(Identity(strong_equality::nonequivalent != 0));
            CHECK(Identity(0 != strong_equality::nonequivalent));
            const strong_equality values[] = {strong_equality::equal,
                                              strong_equality::nonequal};
            for (const auto &lhs: values) {
                for (const auto &rhs: values) {
                    const bool are_equal = &lhs == &rhs;
                    CHECK_EQ(lhs == rhs, are_equal);
                    CHECK_EQ(lhs != rhs, !are_equal);
                }
            }
            CHECK(Identity(strong_equality::equivalent == strong_equality::equal));
            CHECK(
                    Identity(strong_equality::nonequivalent == strong_equality::nonequal));
        }

        TEST_CASE("Compare, PartialOrdering") {
            CHECK(Identity(partial_ordering::less < 0));
            CHECK(Identity(0 > partial_ordering::less));
            CHECK(Identity(partial_ordering::less <= 0));
            CHECK(Identity(0 >= partial_ordering::less));
            CHECK(Identity(partial_ordering::equivalent == 0));
            CHECK(Identity(0 == partial_ordering::equivalent));
            CHECK(Identity(partial_ordering::greater > 0));
            CHECK(Identity(0 < partial_ordering::greater));
            CHECK(Identity(partial_ordering::greater >= 0));
            CHECK(Identity(0 <= partial_ordering::greater));
            CHECK(Identity(partial_ordering::unordered != 0));
            CHECK(Identity(0 != partial_ordering::unordered));
            CHECK_FALSE(Identity(partial_ordering::unordered < 0));
            CHECK_FALSE(Identity(0 < partial_ordering::unordered));
            CHECK_FALSE(Identity(partial_ordering::unordered <= 0));
            CHECK_FALSE(Identity(0 <= partial_ordering::unordered));
            CHECK_FALSE(Identity(partial_ordering::unordered > 0));
            CHECK_FALSE(Identity(0 > partial_ordering::unordered));
            CHECK_FALSE(Identity(partial_ordering::unordered >= 0));
            CHECK_FALSE(Identity(0 >= partial_ordering::unordered));
            const partial_ordering values[] = {
                    partial_ordering::less, partial_ordering::equivalent,
                    partial_ordering::greater, partial_ordering::unordered};
            for (const auto &lhs: values) {
                for (const auto &rhs: values) {
                    const bool are_equal = &lhs == &rhs;
                    CHECK_EQ(lhs == rhs, are_equal);
                    CHECK_EQ(lhs != rhs, !are_equal);
                }
            }
        }

        TEST_CASE("Compare, WeakOrdering") {
            CHECK(Identity(weak_ordering::less < 0));
            CHECK(Identity(0 > weak_ordering::less));
            CHECK(Identity(weak_ordering::less <= 0));
            CHECK(Identity(0 >= weak_ordering::less));
            CHECK(Identity(weak_ordering::equivalent == 0));
            CHECK(Identity(0 == weak_ordering::equivalent));
            CHECK(Identity(weak_ordering::greater > 0));
            CHECK(Identity(0 < weak_ordering::greater));
            CHECK(Identity(weak_ordering::greater >= 0));
            CHECK(Identity(0 <= weak_ordering::greater));
            const weak_ordering values[] = {
                    weak_ordering::less, weak_ordering::equivalent, weak_ordering::greater};
            for (const auto &lhs: values) {
                for (const auto &rhs: values) {
                    const bool are_equal = &lhs == &rhs;
                    CHECK_EQ(lhs == rhs, are_equal);
                    CHECK_EQ(lhs != rhs, !are_equal);
                }
            }
        }

        TEST_CASE("Compare, StrongOrdering") {
            CHECK(Identity(strong_ordering::less < 0));
            CHECK(Identity(0 > strong_ordering::less));
            CHECK(Identity(strong_ordering::less <= 0));
            CHECK(Identity(0 >= strong_ordering::less));
            CHECK(Identity(strong_ordering::equal == 0));
            CHECK(Identity(0 == strong_ordering::equal));
            CHECK(Identity(strong_ordering::equivalent == 0));
            CHECK(Identity(0 == strong_ordering::equivalent));
            CHECK(Identity(strong_ordering::greater > 0));
            CHECK(Identity(0 < strong_ordering::greater));
            CHECK(Identity(strong_ordering::greater >= 0));
            CHECK(Identity(0 <= strong_ordering::greater));
            const strong_ordering values[] = {
                    strong_ordering::less, strong_ordering::equal, strong_ordering::greater};
            for (const auto &lhs: values) {
                for (const auto &rhs: values) {
                    const bool are_equal = &lhs == &rhs;
                    CHECK_EQ(lhs == rhs, are_equal);
                    CHECK_EQ(lhs != rhs, !are_equal);
                }
            }
            CHECK(Identity(strong_ordering::equivalent == strong_ordering::equal));
        }

        TEST_CASE("Compare, Conversions") {
            CHECK(
                    Identity(implicit_cast<weak_equality>(strong_equality::equal) == 0));
            CHECK(
                    Identity(implicit_cast<weak_equality>(strong_equality::nonequal) != 0));
            CHECK(
                    Identity(implicit_cast<weak_equality>(strong_equality::equivalent) == 0));
            CHECK(Identity(
                    implicit_cast<weak_equality>(strong_equality::nonequivalent) != 0));

            CHECK(
                    Identity(implicit_cast<weak_equality>(partial_ordering::less) != 0));
            CHECK(Identity(
                    implicit_cast<weak_equality>(partial_ordering::equivalent) == 0));
            CHECK(
                    Identity(implicit_cast<weak_equality>(partial_ordering::greater) != 0));
            CHECK(
                    Identity(implicit_cast<weak_equality>(partial_ordering::unordered) != 0));

            CHECK(implicit_cast<weak_equality>(weak_ordering::less) != 0);
            CHECK(
                    Identity(implicit_cast<weak_equality>(weak_ordering::equivalent) == 0));
            CHECK(
                    Identity(implicit_cast<weak_equality>(weak_ordering::greater) != 0));

            CHECK(
                    Identity(implicit_cast<partial_ordering>(weak_ordering::less) != 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(weak_ordering::less) < 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(weak_ordering::less) <= 0));
            CHECK(Identity(
                    implicit_cast<partial_ordering>(weak_ordering::equivalent) == 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(weak_ordering::greater) != 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(weak_ordering::greater) > 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(weak_ordering::greater) >= 0));

            CHECK(
                    Identity(implicit_cast<weak_equality>(strong_ordering::less) != 0));
            CHECK(
                    Identity(implicit_cast<weak_equality>(strong_ordering::equal) == 0));
            CHECK(
                    Identity(implicit_cast<weak_equality>(strong_ordering::equivalent) == 0));
            CHECK(
                    Identity(implicit_cast<weak_equality>(strong_ordering::greater) != 0));

            CHECK(
                    Identity(implicit_cast<strong_equality>(strong_ordering::less) != 0));
            CHECK(
                    Identity(implicit_cast<strong_equality>(strong_ordering::equal) == 0));
            CHECK(Identity(
                    implicit_cast<strong_equality>(strong_ordering::equivalent) == 0));
            CHECK(
                    Identity(implicit_cast<strong_equality>(strong_ordering::greater) != 0));

            CHECK(
                    Identity(implicit_cast<partial_ordering>(strong_ordering::less) != 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(strong_ordering::less) < 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(strong_ordering::less) <= 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(strong_ordering::equal) == 0));
            CHECK(Identity(
                    implicit_cast<partial_ordering>(strong_ordering::equivalent) == 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(strong_ordering::greater) != 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(strong_ordering::greater) > 0));
            CHECK(
                    Identity(implicit_cast<partial_ordering>(strong_ordering::greater) >= 0));

            CHECK(
                    Identity(implicit_cast<weak_ordering>(strong_ordering::less) != 0));
            CHECK(
                    Identity(implicit_cast<weak_ordering>(strong_ordering::less) < 0));
            CHECK(
                    Identity(implicit_cast<weak_ordering>(strong_ordering::less) <= 0));
            CHECK(
                    Identity(implicit_cast<weak_ordering>(strong_ordering::equal) == 0));
            CHECK(
                    Identity(implicit_cast<weak_ordering>(strong_ordering::equivalent) == 0));
            CHECK(
                    Identity(implicit_cast<weak_ordering>(strong_ordering::greater) != 0));
            CHECK(
                    Identity(implicit_cast<weak_ordering>(strong_ordering::greater) > 0));
            CHECK(
                    Identity(implicit_cast<weak_ordering>(strong_ordering::greater) >= 0));
        }

        struct WeakOrderingLess {
            template<typename T>
            turbo::weak_ordering operator()(const T &a, const T &b) const {
                return a < b ? turbo::weak_ordering::less
                             : a == b ? turbo::weak_ordering::equivalent
                                      : turbo::weak_ordering::greater;
            }
        };

        TEST_CASE("CompareResultAsLessThan, SanityTest") {
            CHECK_FALSE(turbo::compare_internal::compare_result_as_less_than(false));
            CHECK(turbo::compare_internal::compare_result_as_less_than(true));

            CHECK(
                    turbo::compare_internal::compare_result_as_less_than(weak_ordering::less));
            CHECK_FALSE(turbo::compare_internal::compare_result_as_less_than(
                    weak_ordering::equivalent));
            CHECK_FALSE(turbo::compare_internal::compare_result_as_less_than(
                    weak_ordering::greater));
        }

        TEST_CASE("DoLessThanComparison, SanityTest") {
            std::less<int> less;
            WeakOrderingLess weak;

            CHECK(turbo::compare_internal::do_less_than_comparison(less, -1, 0));
            CHECK(turbo::compare_internal::do_less_than_comparison(weak, -1, 0));

            CHECK_FALSE(turbo::compare_internal::do_less_than_comparison(less, 10, 10));
            CHECK_FALSE(turbo::compare_internal::do_less_than_comparison(weak, 10, 10));

            CHECK_FALSE(turbo::compare_internal::do_less_than_comparison(less, 10, 5));
            CHECK_FALSE(turbo::compare_internal::do_less_than_comparison(weak, 10, 5));
        }

        TEST_CASE("CompareResultAsOrdering, SanityTest") {
            CHECK(
                    Identity(turbo::compare_internal::compare_result_as_ordering(-1) < 0));
            CHECK_FALSE(
                    Identity(turbo::compare_internal::compare_result_as_ordering(-1) == 0));
            CHECK_FALSE(
                    Identity(turbo::compare_internal::compare_result_as_ordering(-1) > 0));
            CHECK(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::less) < 0));
            CHECK_FALSE(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::less) == 0));
            CHECK_FALSE(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::less) > 0));

            CHECK_FALSE(
                    Identity(turbo::compare_internal::compare_result_as_ordering(0) < 0));
            CHECK(
                    Identity(turbo::compare_internal::compare_result_as_ordering(0) == 0));
            CHECK_FALSE(
                    Identity(turbo::compare_internal::compare_result_as_ordering(0) > 0));
            CHECK_FALSE(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::equivalent) < 0));
            CHECK(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::equivalent) == 0));
            CHECK_FALSE(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::equivalent) > 0));

            CHECK_FALSE(
                    Identity(turbo::compare_internal::compare_result_as_ordering(1) < 0));
            CHECK_FALSE(
                    Identity(turbo::compare_internal::compare_result_as_ordering(1) == 0));
            CHECK(
                    Identity(turbo::compare_internal::compare_result_as_ordering(1) > 0));
            CHECK_FALSE(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::greater) < 0));
            CHECK_FALSE(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::greater) == 0));
            CHECK(Identity(turbo::compare_internal::compare_result_as_ordering(
                    weak_ordering::greater) > 0));
        }

        TEST_CASE("DoThreeWayComparison, SanityTest") {
            std::less<int> less;
            WeakOrderingLess weak;

            CHECK(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, -1, 0) < 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, -1, 0) == 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, -1, 0) > 0));
            CHECK(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, -1, 0) < 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, -1, 0) == 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, -1, 0) > 0));

            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, 10, 10) < 0));
            CHECK(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, 10, 10) == 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, 10, 10) > 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, 10, 10) < 0));
            CHECK(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, 10, 10) == 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, 10, 10) > 0));

            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, 10, 5) < 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, 10, 5) == 0));
            CHECK(Identity(
                    turbo::compare_internal::do_three_way_comparison(less, 10, 5) > 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, 10, 5) < 0));
            CHECK_FALSE(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, 10, 5) == 0));
            CHECK(Identity(
                    turbo::compare_internal::do_three_way_comparison(weak, 10, 5) > 0));
        }

#ifdef __cpp_inline_variables

        TEST_CASE("Compare, StaticAsserts") {
            static_assert(weak_equality::equivalent == 0, "");
            static_assert(weak_equality::nonequivalent != 0, "");

            static_assert(strong_equality::equal == 0, "");
            static_assert(strong_equality::nonequal != 0, "");
            static_assert(strong_equality::equivalent == 0, "");
            static_assert(strong_equality::nonequivalent != 0, "");

            static_assert(partial_ordering::less < 0, "");
            static_assert(partial_ordering::equivalent == 0, "");
            static_assert(partial_ordering::greater > 0, "");
            static_assert(partial_ordering::unordered != 0, "");

            static_assert(weak_ordering::less < 0, "");
            static_assert(weak_ordering::equivalent == 0, "");
            static_assert(weak_ordering::greater > 0, "");

            static_assert(strong_ordering::less < 0, "");
            static_assert(strong_ordering::equal == 0, "");
            static_assert(strong_ordering::equivalent == 0, "");
            static_assert(strong_ordering::greater > 0, "");
        }

#endif  // __cpp_inline_variables

    }  // namespace
}  // namespace turbo
