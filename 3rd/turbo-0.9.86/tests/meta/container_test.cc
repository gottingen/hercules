// Copyright 2020 The Turbo Authors.
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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/meta/container.h"

#include <functional>
#include <initializer_list>
#include <iterator>
#include <list>
#include <memory>
#include <ostream>
#include <random>
#include <set>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <vector>

#include "turbo/base/casts.h"
#include "turbo/memory/memory.h"
#include "turbo/meta/span.h"
#include "turbo/platform/port.h"

namespace {

    class NonMutatingTest {
    public:
        NonMutatingTest() = default;

        ~NonMutatingTest() = default;

        std::unordered_set<int> container_ = {1, 2, 3};
        std::list<int> sequence_ = {1, 2, 3};
        std::vector<int> vector_ = {1, 2, 3};
        int array_[3] = {1, 2, 3};
    };

    struct AccumulateCalls {
        void operator()(int value) { calls.push_back(value); }

        std::vector<int> calls;
    };

    bool Predicate(int value) { return value < 3; }

    bool BinPredicate(int v1, int v2) { return v1 < v2; }

    bool Equals(int v1, int v2) { return v1 == v2; }

    bool IsOdd(int x) { return x % 2 != 0; }

    TEST_CASE_FIXTURE(NonMutatingTest, "Distance") {
        REQUIRE_EQ(container_.size(),
                   static_cast<size_t>(turbo::c_distance(container_)));
        REQUIRE_EQ(sequence_.size(), static_cast<size_t>(turbo::c_distance(sequence_)));
        REQUIRE_EQ(vector_.size(), static_cast<size_t>(turbo::c_distance(vector_)));
        REQUIRE_EQ(TURBO_ARRAY_SIZE(array_),
                   static_cast<size_t>(turbo::c_distance(array_)));

        // Works with a temporary argument.
        REQUIRE_EQ(vector_.size(),
                   static_cast<size_t>(turbo::c_distance(std::vector<int>(vector_))));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "Distance_OverloadedBeginEnd") {
        // Works with classes which have custom ADL-selected overloads of std::begin
        // and std::end.
        std::initializer_list<int> a = {1, 2, 3};
        std::valarray<int> b = {1, 2, 3};
        REQUIRE_EQ(3, turbo::c_distance(a));
        REQUIRE_EQ(3, turbo::c_distance(b));

        // It is assumed that other c_* functions use the same mechanism for
        // ADL-selecting begin/end overloads.
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "ForEach") {
        AccumulateCalls c = turbo::c_for_each(container_, AccumulateCalls());
        // Don't rely on the unordered_set's order.
        std::sort(c.calls.begin(), c.calls.end());
        REQUIRE_EQ(vector_, c.calls);

        // Works with temporary container, too.
        AccumulateCalls c2 =
                turbo::c_for_each(std::unordered_set<int>(container_), AccumulateCalls());
        std::sort(c2.calls.begin(), c2.calls.end());
        REQUIRE_EQ(vector_, c2.calls);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "FindReturnsCorrectType") {
        auto it = turbo::c_find(container_, 3);
        REQUIRE_EQ(3, *it);
        turbo::c_find(turbo::implicit_cast<const std::list<int> &>(sequence_), 3);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "FindIf") { turbo::c_find_if(container_, Predicate); }

    TEST_CASE_FIXTURE(NonMutatingTest, "FindIfNot") {
        turbo::c_find_if_not(container_, Predicate);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "FindEnd") {
        turbo::c_find_end(sequence_, vector_);
        turbo::c_find_end(vector_, sequence_);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "FindEndWithPredicate") {
        turbo::c_find_end(sequence_, vector_, BinPredicate);
        turbo::c_find_end(vector_, sequence_, BinPredicate);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "FindFirstOf") {
        turbo::c_find_first_of(container_, sequence_);
        turbo::c_find_first_of(sequence_, container_);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "FindFirstOfWithPredicate") {
        turbo::c_find_first_of(container_, sequence_, BinPredicate);
        turbo::c_find_first_of(sequence_, container_, BinPredicate);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "AdjacentFind") { turbo::c_adjacent_find(sequence_); }

    TEST_CASE_FIXTURE(NonMutatingTest, "AdjacentFindWithPredicate") {
        turbo::c_adjacent_find(sequence_, BinPredicate);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "Count") { REQUIRE_EQ(1, turbo::c_count(container_, 3)); }

    TEST_CASE_FIXTURE(NonMutatingTest, "CountIf") {
        REQUIRE_EQ(2, turbo::c_count_if(container_, Predicate));
        const std::unordered_set<int> &const_container = container_;
        REQUIRE_EQ(2, turbo::c_count_if(const_container, Predicate));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "Mismatch") {
        // Testing necessary as turbo::c_mismatch executes logic.
        {
            auto result = turbo::c_mismatch(vector_, sequence_);
            REQUIRE_EQ(result.first, vector_.end());
            REQUIRE_EQ(result.second, sequence_.end());
        }
        {
            auto result = turbo::c_mismatch(sequence_, vector_);
            REQUIRE_EQ(result.first, sequence_.end());
            REQUIRE_EQ(result.second, vector_.end());
        }

        sequence_.back() = 5;
        {
            auto result = turbo::c_mismatch(vector_, sequence_);
            REQUIRE_EQ(result.first, std::prev(vector_.end()));
            REQUIRE_EQ(result.second, std::prev(sequence_.end()));
        }
        {
            auto result = turbo::c_mismatch(sequence_, vector_);
            REQUIRE_EQ(result.first, std::prev(sequence_.end()));
            REQUIRE_EQ(result.second, std::prev(vector_.end()));
        }

        sequence_.pop_back();
        {
            auto result = turbo::c_mismatch(vector_, sequence_);
            REQUIRE_EQ(result.first, std::prev(vector_.end()));
            REQUIRE_EQ(result.second, sequence_.end());
        }
        {
            auto result = turbo::c_mismatch(sequence_, vector_);
            REQUIRE_EQ(result.first, sequence_.end());
            REQUIRE_EQ(result.second, std::prev(vector_.end()));
        }
        {
            struct NoNotEquals {
                constexpr bool operator==(NoNotEquals) const { return true; }

                constexpr bool operator!=(NoNotEquals) const = delete;
            };
            std::vector<NoNotEquals> first;
            std::list<NoNotEquals> second;

            // Check this still compiles.
            turbo::c_mismatch(first, second);
        }
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "MismatchWithPredicate") {
        // Testing necessary as turbo::c_mismatch executes logic.
        {
            auto result = turbo::c_mismatch(vector_, sequence_, BinPredicate);
            REQUIRE_EQ(result.first, vector_.begin());
            REQUIRE_EQ(result.second, sequence_.begin());
        }
        {
            auto result = turbo::c_mismatch(sequence_, vector_, BinPredicate);
            REQUIRE_EQ(result.first, sequence_.begin());
            REQUIRE_EQ(result.second, vector_.begin());
        }

        sequence_.front() = 0;
        {
            auto result = turbo::c_mismatch(vector_, sequence_, BinPredicate);
            REQUIRE_EQ(result.first, vector_.begin());
            REQUIRE_EQ(result.second, sequence_.begin());
        }
        {
            auto result = turbo::c_mismatch(sequence_, vector_, BinPredicate);
            REQUIRE_EQ(result.first, std::next(sequence_.begin()));
            REQUIRE_EQ(result.second, std::next(vector_.begin()));
        }

        sequence_.clear();
        {
            auto result = turbo::c_mismatch(vector_, sequence_, BinPredicate);
            REQUIRE_EQ(result.first, vector_.begin());
            REQUIRE_EQ(result.second, sequence_.end());
        }
        {
            auto result = turbo::c_mismatch(sequence_, vector_, BinPredicate);
            REQUIRE_EQ(result.first, sequence_.end());
            REQUIRE_EQ(result.second, vector_.begin());
        }
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "Equal") {
        REQUIRE(turbo::c_equal(vector_, sequence_));
        REQUIRE(turbo::c_equal(sequence_, vector_));
        REQUIRE(turbo::c_equal(sequence_, array_));
        REQUIRE(turbo::c_equal(array_, vector_));

        // Test that behavior appropriately differs from that of equal().
        std::vector<int> vector_plus = {1, 2, 3};
        vector_plus.push_back(4);
        REQUIRE_FALSE(turbo::c_equal(vector_plus, sequence_));
        REQUIRE_FALSE(turbo::c_equal(sequence_, vector_plus));
        REQUIRE_FALSE(turbo::c_equal(array_, vector_plus));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "EqualWithPredicate") {
        REQUIRE(turbo::c_equal(vector_, sequence_, Equals));
        REQUIRE(turbo::c_equal(sequence_, vector_, Equals));
        REQUIRE(turbo::c_equal(array_, sequence_, Equals));
        REQUIRE(turbo::c_equal(vector_, array_, Equals));

        // Test that behavior appropriately differs from that of equal().
        std::vector<int> vector_plus = {1, 2, 3};
        vector_plus.push_back(4);
        REQUIRE_FALSE(turbo::c_equal(vector_plus, sequence_, Equals));
        REQUIRE_FALSE(turbo::c_equal(sequence_, vector_plus, Equals));
        REQUIRE_FALSE(turbo::c_equal(vector_plus, array_, Equals));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "IsPermutation") {
        auto vector_permut_ = vector_;
        std::next_permutation(vector_permut_.begin(), vector_permut_.end());
        REQUIRE(turbo::c_is_permutation(vector_permut_, sequence_));
        REQUIRE(turbo::c_is_permutation(sequence_, vector_permut_));

        // Test that behavior appropriately differs from that of is_permutation().
        std::vector<int> vector_plus = {1, 2, 3};
        vector_plus.push_back(4);
        REQUIRE_FALSE(turbo::c_is_permutation(vector_plus, sequence_));
        REQUIRE_FALSE(turbo::c_is_permutation(sequence_, vector_plus));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "IsPermutationWithPredicate") {
        auto vector_permut_ = vector_;
        std::next_permutation(vector_permut_.begin(), vector_permut_.end());
        REQUIRE(turbo::c_is_permutation(vector_permut_, sequence_, Equals));
        REQUIRE(turbo::c_is_permutation(sequence_, vector_permut_, Equals));

        // Test that behavior appropriately differs from that of is_permutation().
        std::vector<int> vector_plus = {1, 2, 3};
        vector_plus.push_back(4);
        REQUIRE_FALSE(turbo::c_is_permutation(vector_plus, sequence_, Equals));
        REQUIRE_FALSE(turbo::c_is_permutation(sequence_, vector_plus, Equals));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "Search") {
        turbo::c_search(sequence_, vector_);
        turbo::c_search(vector_, sequence_);
        turbo::c_search(array_, sequence_);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "SearchWithPredicate") {
        turbo::c_search(sequence_, vector_, BinPredicate);
        turbo::c_search(vector_, sequence_, BinPredicate);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "SearchN") { turbo::c_search_n(sequence_, 3, 1); }

    TEST_CASE_FIXTURE(NonMutatingTest, "SearchNWithPredicate") {
        turbo::c_search_n(sequence_, 3, 1, BinPredicate);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "LowerBound") {
        std::list<int>::iterator i = turbo::c_lower_bound(sequence_, 3);
        REQUIRE(i != sequence_.end());
        REQUIRE_EQ(2, std::distance(sequence_.begin(), i));
        REQUIRE_EQ(3, *i);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "LowerBoundWithPredicate") {
        std::vector<int> v(vector_);
        std::sort(v.begin(), v.end(), std::greater<int>());
        std::vector<int>::iterator i = turbo::c_lower_bound(v, 3, std::greater<int>());
        REQUIRE(i == v.begin());
        REQUIRE_EQ(3, *i);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "UpperBound") {
        std::list<int>::iterator i = turbo::c_upper_bound(sequence_, 1);
        REQUIRE(i != sequence_.end());
        REQUIRE_EQ(1, std::distance(sequence_.begin(), i));
        REQUIRE_EQ(2, *i);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "UpperBoundWithPredicate") {
        std::vector<int> v(vector_);
        std::sort(v.begin(), v.end(), std::greater<int>());
        std::vector<int>::iterator i = turbo::c_upper_bound(v, 1, std::greater<int>());
        REQUIRE_EQ(3, i - v.begin());
        REQUIRE(i == v.end());
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "EqualRange") {
        std::pair<std::list<int>::iterator, std::list<int>::iterator> p =
                turbo::c_equal_range(sequence_, 2);
        REQUIRE_EQ(1, std::distance(sequence_.begin(), p.first));
        REQUIRE_EQ(2, std::distance(sequence_.begin(), p.second));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "EqualRangeArray") {
        auto p = turbo::c_equal_range(array_, 2);
        REQUIRE_EQ(1, std::distance(std::begin(array_), p.first));
        REQUIRE_EQ(2, std::distance(std::begin(array_), p.second));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "EqualRangeWithPredicate") {
        std::vector<int> v(vector_);
        std::sort(v.begin(), v.end(), std::greater<int>());
        std::pair<std::vector<int>::iterator, std::vector<int>::iterator> p =
                turbo::c_equal_range(v, 2, std::greater<int>());
        REQUIRE_EQ(1, std::distance(v.begin(), p.first));
        REQUIRE_EQ(2, std::distance(v.begin(), p.second));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "BinarySearch") {
        REQUIRE(turbo::c_binary_search(vector_, 2));
        REQUIRE(turbo::c_binary_search(std::vector<int>(vector_), 2));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "BinarySearchWithPredicate") {
        std::vector<int> v(vector_);
        std::sort(v.begin(), v.end(), std::greater<int>());
        REQUIRE(turbo::c_binary_search(v, 2, std::greater<int>()));
        REQUIRE(
                turbo::c_binary_search(std::vector<int>(v), 2, std::greater<int>()));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "MinElement") {
        std::list<int>::iterator i = turbo::c_min_element(sequence_);
        REQUIRE(i != sequence_.end());
        REQUIRE_EQ(*i, 1);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "MinElementWithPredicate") {
        std::list<int>::iterator i =
                turbo::c_min_element(sequence_, std::greater<int>());
        REQUIRE(i != sequence_.end());
        REQUIRE_EQ(*i, 3);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "MaxElement") {
        std::list<int>::iterator i = turbo::c_max_element(sequence_);
        REQUIRE(i != sequence_.end());
        REQUIRE_EQ(*i, 3);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "MaxElementWithPredicate") {
        std::list<int>::iterator i =
                turbo::c_max_element(sequence_, std::greater<int>());
        REQUIRE(i != sequence_.end());
        REQUIRE_EQ(*i, 1);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "LexicographicalCompare") {
        REQUIRE_FALSE(turbo::c_lexicographical_compare(sequence_, sequence_));

        std::vector<int> v;
        v.push_back(1);
        v.push_back(2);
        v.push_back(4);

        REQUIRE(turbo::c_lexicographical_compare(sequence_, v));
        REQUIRE(turbo::c_lexicographical_compare(std::list<int>(sequence_), v));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "LexicographicalCopmareWithPredicate") {
        REQUIRE_FALSE(turbo::c_lexicographical_compare(sequence_, sequence_,
                                                       std::greater<int>()));

        std::vector<int> v;
        v.push_back(1);
        v.push_back(2);
        v.push_back(4);

        REQUIRE(
                turbo::c_lexicographical_compare(v, sequence_, std::greater<int>()));
        REQUIRE(turbo::c_lexicographical_compare(
                std::vector<int>(v), std::list<int>(sequence_), std::greater<int>()));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "Includes") {
        std::set<int> s(vector_.begin(), vector_.end());
        s.insert(4);
        REQUIRE(turbo::c_includes(s, vector_));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "IncludesWithPredicate") {
        std::vector<int> v = {3, 2, 1};
        std::set<int, std::greater<int>> s(v.begin(), v.end());
        s.insert(4);
        REQUIRE(turbo::c_includes(s, v, std::greater<int>()));
    }

    class NumericMutatingTest {
    public:
        NumericMutatingTest() = default;

        ~NumericMutatingTest() = default;

        std::list<int> list_ = {1, 2, 3};
        std::vector<int> output_;
    };

    TEST_CASE_FIXTURE(NumericMutatingTest, "Iota") {
        turbo::c_iota(list_, 5);
        std::list<int> expected{5, 6, 7};
        REQUIRE_EQ(list_, expected);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "Accumulate") {
        REQUIRE_EQ(turbo::c_accumulate(sequence_, 4), 1 + 2 + 3 + 4);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "AccumulateWithBinaryOp") {
        REQUIRE_EQ(turbo::c_accumulate(sequence_, 4, std::multiplies<int>()),
                   1 * 2 * 3 * 4);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "AccumulateLvalueInit") {
        int lvalue = 4;
        REQUIRE_EQ(turbo::c_accumulate(sequence_, lvalue), 1 + 2 + 3 + 4);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "AccumulateWithBinaryOpLvalueInit") {
        int lvalue = 4;
        REQUIRE_EQ(turbo::c_accumulate(sequence_, lvalue, std::multiplies<int>()),
                   1 * 2 * 3 * 4);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "InnerProduct") {
        REQUIRE_EQ(turbo::c_inner_product(sequence_, vector_, 1000),
                   1000 + 1 * 1 + 2 * 2 + 3 * 3);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "InnerProductWithBinaryOps") {
        REQUIRE_EQ(turbo::c_inner_product(sequence_, vector_, 10,
                                          std::multiplies<int>(), std::plus<int>()),
                   10 * (1 + 1) * (2 + 2) * (3 + 3));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "InnerProductLvalueInit") {
        int lvalue = 1000;
        REQUIRE_EQ(turbo::c_inner_product(sequence_, vector_, lvalue),
                   1000 + 1 * 1 + 2 * 2 + 3 * 3);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "InnerProductWithBinaryOpsLvalueInit") {
        int lvalue = 10;
        REQUIRE_EQ(turbo::c_inner_product(sequence_, vector_, lvalue,
                                          std::multiplies<int>(), std::plus<int>()),
                   10 * (1 + 1) * (2 + 2) * (3 + 3));
    }

    TEST_CASE_FIXTURE(NumericMutatingTest, "AdjacentDifference") {
        auto last = turbo::c_adjacent_difference(list_, std::back_inserter(output_));
        *last = 1000;
        std::vector<int> expected{1, 2 - 1, 3 - 2, 1000};
        REQUIRE_EQ(output_, expected);
    }

    TEST_CASE_FIXTURE(NumericMutatingTest, "AdjacentDifferenceWithBinaryOp") {
        auto last = turbo::c_adjacent_difference(list_, std::back_inserter(output_),
                                                 std::multiplies<int>());
        *last = 1000;
        std::vector<int> expected{1, 2 * 1, 3 * 2, 1000};
        REQUIRE_EQ(output_, expected);
    }

    TEST_CASE_FIXTURE(NumericMutatingTest, "PartialSum") {
        auto last = turbo::c_partial_sum(list_, std::back_inserter(output_));
        *last = 1000;
        std::vector<int> expected{1, 1 + 2, 1 + 2 + 3, 1000};
        REQUIRE_EQ(output_, expected);
    }

    TEST_CASE_FIXTURE(NumericMutatingTest, "PartialSumWithBinaryOp") {
        auto last = turbo::c_partial_sum(list_, std::back_inserter(output_),
                                         std::multiplies<int>());
        *last = 1000;
        std::vector<int> expected{1, 1 * 2, 1 * 2 * 3, 1000};
        REQUIRE_EQ(output_, expected);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "LinearSearch") {
        REQUIRE(turbo::c_linear_search(container_, 3));
        REQUIRE_FALSE(turbo::c_linear_search(container_, 4));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "AllOf") {
        const std::vector<int> &v = vector_;
        REQUIRE_FALSE(turbo::c_all_of(v, [](int x) { return x > 1; }));
        REQUIRE(turbo::c_all_of(v, [](int x) { return x > 0; }));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "AnyOf") {
        const std::vector<int> &v = vector_;
        REQUIRE(turbo::c_any_of(v, [](int x) { return x > 2; }));
        REQUIRE_FALSE(turbo::c_any_of(v, [](int x) { return x > 5; }));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "NoneOf") {
        const std::vector<int> &v = vector_;
        REQUIRE_FALSE(turbo::c_none_of(v, [](int x) { return x > 2; }));
        REQUIRE(turbo::c_none_of(v, [](int x) { return x > 5; }));
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "MinMaxElementLess") {
        std::pair<std::vector<int>::const_iterator, std::vector<int>::const_iterator>
                p = turbo::c_minmax_element(vector_, std::less<int>());
        REQUIRE(p.first == vector_.begin());
        REQUIRE(p.second == vector_.begin() + 2);
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "MinMaxElementGreater") {
        std::pair<std::vector<int>::const_iterator, std::vector<int>::const_iterator>
                p = turbo::c_minmax_element(vector_, std::greater<int>());
        REQUIRE(p.first == vector_.begin() + 2);
        REQUIRE(p.second == vector_.begin());
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "MinMaxElementNoPredicate") {
        std::pair<std::vector<int>::const_iterator, std::vector<int>::const_iterator>
                p = turbo::c_minmax_element(vector_);
        REQUIRE(p.first == vector_.begin());
        REQUIRE(p.second == vector_.begin() + 2);
    }

    class SortingTest {
    public:
        SortingTest() = default;

        ~SortingTest() = default;

        std::list<int> sorted_ = {1, 2, 3, 4};
        std::list<int> unsorted_ = {2, 4, 1, 3};
        std::list<int> reversed_ = {4, 3, 2, 1};
    };

    TEST_CASE_FIXTURE(SortingTest, "IsSorted") {
        REQUIRE(turbo::c_is_sorted(sorted_));
        REQUIRE_FALSE(turbo::c_is_sorted(unsorted_));
        REQUIRE_FALSE(turbo::c_is_sorted(reversed_));
    }

    TEST_CASE_FIXTURE(SortingTest, "IsSortedWithPredicate") {
        REQUIRE_FALSE(turbo::c_is_sorted(sorted_, std::greater<int>()));
        REQUIRE_FALSE(turbo::c_is_sorted(unsorted_, std::greater<int>()));
        REQUIRE(turbo::c_is_sorted(reversed_, std::greater<int>()));
    }

    TEST_CASE_FIXTURE(SortingTest, "IsSortedUntil") {
        REQUIRE_EQ(1, *turbo::c_is_sorted_until(unsorted_));
        REQUIRE_EQ(4, *turbo::c_is_sorted_until(unsorted_, std::greater<int>()));
    }


    TEST_CASE("MutatingTest, IsPartitioned") {
        REQUIRE(
                turbo::c_is_partitioned(std::vector<int>{1, 3, 5, 2, 4, 6}, IsOdd));
        REQUIRE_FALSE(
                turbo::c_is_partitioned(std::vector<int>{1, 2, 3, 4, 5, 6}, IsOdd));
        REQUIRE_FALSE(
                turbo::c_is_partitioned(std::vector<int>{2, 4, 6, 1, 3, 5}, IsOdd));
    }


    TEST_CASE("MutatingTest, StablePartition") {
        std::vector<int> actual = {1, 2, 3, 4, 5};
        turbo::c_stable_partition(actual, IsOdd);
        REQUIRE_EQ(actual, std::vector<int>{1, 3, 5, 2, 4});
    }

    TEST_CASE("MutatingTest, PartitionCopy") {
        const std::vector<int> initial = {1, 2, 3, 4, 5};
        std::vector<int> odds, evens;
        auto ends = turbo::c_partition_copy(initial, back_inserter(odds),
                                            back_inserter(evens), IsOdd);
        *ends.first = 7;
        *ends.second = 6;
        REQUIRE_EQ(odds, std::vector<int>{1, 3, 5, 7});
        REQUIRE_EQ(evens, std::vector<int>{2, 4, 6});
    }

    TEST_CASE("MutatingTest, PartitionPoint") {
        const std::vector<int> initial = {1, 3, 5, 2, 4};
        auto middle = turbo::c_partition_point(initial, IsOdd);
        REQUIRE_EQ(2, *middle);
    }

    TEST_CASE("MutatingTest, CopyMiddle") {
        const std::vector<int> initial = {4, -1, -2, -3, 5};
        const std::list<int> input = {1, 2, 3};
        const std::vector<int> expected = {4, 1, 2, 3, 5};

        std::list<int> test_list(initial.begin(), initial.end());
        turbo::c_copy(input, ++test_list.begin());
        REQUIRE_EQ(std::list<int>(expected.begin(), expected.end()), test_list);

        std::vector<int> test_vector = initial;
        turbo::c_copy(input, test_vector.begin() + 1);
        REQUIRE_EQ(expected, test_vector);
    }

    TEST_CASE("MutatingTest, CopyFrontInserter") {
        const std::list<int> initial = {4, 5};
        const std::list<int> input = {1, 2, 3};
        const std::list<int> expected = {3, 2, 1, 4, 5};

        std::list<int> test_list = initial;
        turbo::c_copy(input, std::front_inserter(test_list));
        REQUIRE_EQ(expected, test_list);
    }

    TEST_CASE("MutatingTest, CopyBackInserter") {
        const std::vector<int> initial = {4, 5};
        const std::list<int> input = {1, 2, 3};
        const std::vector<int> expected = {4, 5, 1, 2, 3};

        std::list<int> test_list(initial.begin(), initial.end());
        turbo::c_copy(input, std::back_inserter(test_list));
        REQUIRE_EQ(std::list<int>(expected.begin(), expected.end()), test_list);

        std::vector<int> test_vector = initial;
        turbo::c_copy(input, std::back_inserter(test_vector));
        REQUIRE_EQ(expected, test_vector);
    }

    TEST_CASE("MutatingTest, CopyN") {
        const std::vector<int> initial = {1, 2, 3, 4, 5};
        const std::vector<int> expected = {1, 2};
        std::vector<int> actual;
        turbo::c_copy_n(initial, 2, back_inserter(actual));
        REQUIRE_EQ(expected, actual);
    }

    TEST_CASE("MutatingTest, CopyIf") {
        const std::list<int> input = {1, 2, 3};
        std::vector<int> output;
        turbo::c_copy_if(input, std::back_inserter(output),
                         [](int i) { return i != 2; });
        REQUIRE_EQ(output, std::vector<int>{1, 3});
    }

    TEST_CASE("MutatingTest, CopyBackward") {
        std::vector<int> actual = {1, 2, 3, 4, 5};
        std::vector<int> expected = {1, 2, 1, 2, 3};
        turbo::c_copy_backward(turbo::MakeSpan(actual.data(), 3), actual.end());
        REQUIRE_EQ(expected, actual);
    }


    TEST_CASE("MutatingTest, SwapRanges") {
        std::vector<int> odds = {2, 4, 6};
        std::vector<int> evens = {1, 3, 5};
        turbo::c_swap_ranges(odds, evens);
        REQUIRE_EQ(odds, std::vector<int>{1, 3, 5});
        REQUIRE_EQ(evens, std::vector<int>{2, 4, 6});

        odds.pop_back();
        turbo::c_swap_ranges(odds, evens);
        REQUIRE_EQ(odds, std::vector<int>{2, 4});
        REQUIRE_EQ(evens, std::vector<int>{1, 3, 6});

        turbo::c_swap_ranges(evens, odds);
        REQUIRE_EQ(odds, std::vector<int>{1, 3});
        REQUIRE_EQ(evens, std::vector<int>{2, 4, 6});
    }

    TEST_CASE_FIXTURE(NonMutatingTest, "Transform") {
        std::vector<int> x{0, 2, 4}, y, z;
        auto end = turbo::c_transform(x, back_inserter(y), std::negate<int>());
        REQUIRE_EQ(std::vector<int>({0, -2, -4}), y);
        *end = 7;
        REQUIRE_EQ(std::vector<int>({0, -2, -4, 7}), y);

        y = {1, 3, 0};
        end = turbo::c_transform(x, y, back_inserter(z), std::plus<int>());
        REQUIRE_EQ(std::vector<int>({1, 5, 4}), z);
        *end = 7;
        REQUIRE_EQ(std::vector<int>({1, 5, 4, 7}), z);

        z.clear();
        y.pop_back();
        end = turbo::c_transform(x, y, std::back_inserter(z), std::plus<int>());
        REQUIRE_EQ(std::vector<int>({1, 5}), z);
        *end = 7;
        REQUIRE_EQ(std::vector<int>({1, 5, 7}), z);

        z.clear();
        std::swap(x, y);
        end = turbo::c_transform(x, y, std::back_inserter(z), std::plus<int>());
        REQUIRE_EQ(std::vector<int>({1, 5}), z);
        *end = 7;
        REQUIRE_EQ(std::vector<int>({1, 5, 7}), z);
    }

    TEST_CASE("MutatingTest, Replace") {
        const std::vector<int> initial = {1, 2, 3, 1, 4, 5};
        const std::vector<int> expected = {4, 2, 3, 4, 4, 5};

        std::vector<int> test_vector = initial;
        turbo::c_replace(test_vector, 1, 4);
        REQUIRE_EQ(expected, test_vector);

        std::list<int> test_list(initial.begin(), initial.end());
        turbo::c_replace(test_list, 1, 4);
        REQUIRE_EQ(std::list<int>(expected.begin(), expected.end()), test_list);
    }

    TEST_CASE("MutatingTest, ReplaceIf") {
        std::vector<int> actual = {1, 2, 3, 4, 5};
        const std::vector<int> expected = {0, 2, 0, 4, 0};

        turbo::c_replace_if(actual, IsOdd, 0);
        REQUIRE_EQ(expected, actual);
    }

    TEST_CASE("MutatingTest, ReplaceCopy") {
        const std::vector<int> initial = {1, 2, 3, 1, 4, 5};
        const std::vector<int> expected = {4, 2, 3, 4, 4, 5};

        std::vector<int> actual;
        turbo::c_replace_copy(initial, back_inserter(actual), 1, 4);
        REQUIRE_EQ(expected, actual);
    }

    TEST_CASE("MutatingTest, Sort") {
        std::vector<int> test_vector = {2, 3, 1, 4};
        turbo::c_sort(test_vector);
        REQUIRE_EQ(test_vector, std::vector<int>{1, 2, 3, 4});
    }

    TEST_CASE("MutatingTest, SortWithPredicate") {
        std::vector<int> test_vector = {2, 3, 1, 4};
        turbo::c_sort(test_vector, std::greater<int>());
        REQUIRE_EQ(test_vector, std::vector<int>{4, 3, 2, 1});
    }


    TEST_CASE("MutatingTest, ReplaceCopyIf") {
        const std::vector<int> initial = {1, 2, 3, 4, 5};
        const std::vector<int> expected = {0, 2, 0, 4, 0};

        std::vector<int> actual;
        turbo::c_replace_copy_if(initial, back_inserter(actual), IsOdd, 0);
        REQUIRE_EQ(expected, actual);
    }

    TEST_CASE("MutatingTest, Fill") {
        std::vector<int> actual(5);
        turbo::c_fill(actual, 1);
        REQUIRE_EQ(actual, std::vector<int>{1, 1, 1, 1, 1});
    }

    TEST_CASE("MutatingTest, FillN") {
        std::vector<int> actual(5, 0);
        turbo::c_fill_n(actual, 2, 1);
        REQUIRE_EQ(actual, std::vector<int>{1, 1, 0, 0, 0});
    }

    TEST_CASE("MutatingTest, Generate") {
        std::vector<int> actual(5);
        int x = 0;
        turbo::c_generate(actual, [&x]() { return ++x; });
        REQUIRE_EQ(actual, std::vector<int>{1, 2, 3, 4, 5});
    }

    TEST_CASE("MutatingTest, GenerateN") {
        std::vector<int> actual(5, 0);
        int x = 0;
        turbo::c_generate_n(actual, 3, [&x]() { return ++x; });
        REQUIRE_EQ(actual, std::vector<int>{1, 2, 3, 0, 0});
    }

    TEST_CASE("MutatingTest, RemoveCopy") {
        std::vector<int> actual;
        turbo::c_remove_copy(std::vector<int>{1, 2, 3}, back_inserter(actual), 2);
        REQUIRE_EQ(actual, std::vector<int>{1, 3});
    }

    TEST_CASE("MutatingTest, RemoveCopyIf") {
        std::vector<int> actual;
        turbo::c_remove_copy_if(std::vector<int>{1, 2, 3}, back_inserter(actual),
                                IsOdd);
        REQUIRE_EQ(actual, std::vector<int>{2});
    }

    TEST_CASE("MutatingTest, UniqueCopy") {
        std::vector<int> actual;
        turbo::c_unique_copy(std::vector<int>{1, 2, 2, 2, 3, 3, 2},
                             back_inserter(actual));
        REQUIRE_EQ(actual, std::vector<int>{1, 2, 3, 2});
    }

    TEST_CASE("MutatingTest, UniqueCopyWithPredicate") {
        std::vector<int> actual;
        turbo::c_unique_copy(std::vector<int>{1, 2, 3, -1, -2, -3, 1},
                             back_inserter(actual),
                             [](int x, int y) { return (x < 0) == (y < 0); });
        REQUIRE_EQ(actual, std::vector<int>{1, -1, 1});
    }

    TEST_CASE("MutatingTest, Reverse") {
        std::vector<int> test_vector = {1, 2, 3, 4};
        turbo::c_reverse(test_vector);
        REQUIRE_EQ(test_vector, std::vector<int>{4, 3, 2, 1});

        std::list<int> test_list = {1, 2, 3, 4};
        turbo::c_reverse(test_list);
        REQUIRE_EQ(test_list, std::list<int>{4, 3, 2, 1});
    }

    TEST_CASE("MutatingTest, ReverseCopy") {
        std::vector<int> actual;
        turbo::c_reverse_copy(std::vector<int>{1, 2, 3, 4}, back_inserter(actual));
        REQUIRE_EQ(actual, std::vector<int>{4, 3, 2, 1});
    }

    TEST_CASE("MutatingTest, Rotate") {
        std::vector<int> actual = {1, 2, 3, 4};
        auto it = turbo::c_rotate(actual, actual.begin() + 2);
        REQUIRE_EQ(actual, std::vector<int>{3, 4, 1, 2});
        REQUIRE_EQ(*it, 1);
    }

    TEST_CASE("MutatingTest, RotateCopy") {
        std::vector<int> initial = {1, 2, 3, 4};
        std::vector<int> actual;
        auto end =
                turbo::c_rotate_copy(initial, initial.begin() + 2, back_inserter(actual));
        *end = 5;
        REQUIRE_EQ(actual, std::vector<int>{3, 4, 1, 2, 5});
    }


    TEST_CASE("MutatingTest, PartialSort") {
        std::vector<int> sequence{5, 3, 42, 0};
        turbo::c_partial_sort(sequence, sequence.begin() + 2);
        REQUIRE_EQ(turbo::MakeSpan(sequence.data(), 2), std::vector<int>{0, 3});
        turbo::c_partial_sort(sequence, sequence.begin() + 2, std::greater<int>());
        REQUIRE_EQ(turbo::MakeSpan(sequence.data(), 2), std::vector<int>{42, 5});
    }

    TEST_CASE("MutatingTest, PartialSortCopy") {
        const std::vector<int> initial = {5, 3, 42, 0};
        std::vector<int> actual(2);
        turbo::c_partial_sort_copy(initial, actual);
        REQUIRE_EQ(actual, std::vector<int>{0, 3});
        turbo::c_partial_sort_copy(initial, actual, std::greater<int>());
        REQUIRE_EQ(actual, std::vector<int>{42, 5});
    }

    TEST_CASE("MutatingTest, Merge") {
        std::vector<int> actual;
        turbo::c_merge(std::vector<int>{1, 3, 5}, std::vector<int>{2, 4},
                       back_inserter(actual));
        REQUIRE_EQ(actual, std::vector<int>{1, 2, 3, 4, 5});
    }

    TEST_CASE("MutatingTest, MergeWithComparator") {
        std::vector<int> actual;
        turbo::c_merge(std::vector<int>{5, 3, 1}, std::vector<int>{4, 2},
                       back_inserter(actual), std::greater<int>());
        REQUIRE_EQ(actual, std::vector<int>{5, 4, 3, 2, 1});
    }

    TEST_CASE("MutatingTest, InplaceMerge") {
        std::vector<int> actual = {1, 3, 5, 2, 4};
        turbo::c_inplace_merge(actual, actual.begin() + 3);
        REQUIRE_EQ(actual, std::vector<int>{1, 2, 3, 4, 5});
    }

    TEST_CASE("MutatingTest, InplaceMergeWithComparator") {
        std::vector<int> actual = {5, 3, 1, 4, 2};
        turbo::c_inplace_merge(actual, actual.begin() + 3, std::greater<int>());
        REQUIRE_EQ(actual, std::vector<int>{5, 4, 3, 2, 1});
    }

    class SetOperationsTest {
    public:
        SetOperationsTest() = default;

        ~SetOperationsTest() = default;

        std::vector<int> a_ = {1, 2, 3};
        std::vector<int> b_ = {1, 3, 5};

        std::vector<int> a_reversed_ = {3, 2, 1};
        std::vector<int> b_reversed_ = {5, 3, 1};
    };

    TEST_CASE_FIXTURE(SetOperationsTest, "SetUnion") {
        std::vector<int> actual;
        turbo::c_set_union(a_, b_, back_inserter(actual));
        REQUIRE_EQ(actual, std::vector<int>{1, 2, 3, 5});
    }

    TEST_CASE_FIXTURE(SetOperationsTest, "SetUnionWithComparator") {
        std::vector<int> actual;
        turbo::c_set_union(a_reversed_, b_reversed_, back_inserter(actual),
                           std::greater<int>());
        REQUIRE_EQ(actual, std::vector<int>{5, 3, 2, 1});
    }

    TEST_CASE_FIXTURE(SetOperationsTest, "SetIntersection") {
        std::vector<int> actual;
        turbo::c_set_intersection(a_, b_, back_inserter(actual));
        REQUIRE_EQ(actual, std::vector<int>{1, 3});
    }

    TEST_CASE_FIXTURE(SetOperationsTest, "SetIntersectionWithComparator") {
        std::vector<int> actual;
        turbo::c_set_intersection(a_reversed_, b_reversed_, back_inserter(actual),
                                  std::greater<int>());
        REQUIRE_EQ(actual, std::vector<int>{3, 1});
    }

    TEST_CASE_FIXTURE(SetOperationsTest, "SetDifference") {
        std::vector<int> actual;
        turbo::c_set_difference(a_, b_, back_inserter(actual));
        REQUIRE_EQ(actual, std::vector<int>{2});
    }

    TEST_CASE_FIXTURE(SetOperationsTest, "SetDifferenceWithComparator") {
        std::vector<int> actual;
        turbo::c_set_difference(a_reversed_, b_reversed_, back_inserter(actual),
                                std::greater<int>());
        REQUIRE_EQ(actual, std::vector<int>{2});
    }

    TEST_CASE_FIXTURE(SetOperationsTest, "SetSymmetricDifference") {
        std::vector<int> actual;
        turbo::c_set_symmetric_difference(a_, b_, back_inserter(actual));
        REQUIRE_EQ(actual, std::vector<int>{2, 5});
    }

    TEST_CASE_FIXTURE(SetOperationsTest, "SetSymmetricDifferenceWithComparator") {
        std::vector<int> actual;
        turbo::c_set_symmetric_difference(a_reversed_, b_reversed_,
                                          back_inserter(actual), std::greater<int>());
        REQUIRE_EQ(actual, std::vector<int>{5, 2});
    }

    TEST_CASE("HeapOperationsTest, WithoutComparator") {
        std::vector<int> heap = {1, 2, 3};
        REQUIRE_FALSE(turbo::c_is_heap(heap));
        turbo::c_make_heap(heap);
        REQUIRE(turbo::c_is_heap(heap));
        heap.push_back(4);
        REQUIRE_EQ(3, turbo::c_is_heap_until(heap) - heap.begin());
        turbo::c_push_heap(heap);
        REQUIRE_EQ(4, heap[0]);
        turbo::c_pop_heap(heap);
        REQUIRE_EQ(4, heap[3]);
        turbo::c_make_heap(heap);
        turbo::c_sort_heap(heap);
        REQUIRE_EQ(heap, std::vector<int>{1, 2, 3, 4});
        REQUIRE_FALSE(turbo::c_is_heap(heap));
    }

    TEST_CASE("HeapOperationsTest, WithComparator") {
        using greater = std::greater<int>;
        std::vector<int> heap = {3, 2, 1};
        REQUIRE_FALSE(turbo::c_is_heap(heap, greater()));
        turbo::c_make_heap(heap, greater());
        REQUIRE(turbo::c_is_heap(heap, greater()));
        heap.push_back(0);
        REQUIRE_EQ(3, turbo::c_is_heap_until(heap, greater()) - heap.begin());
        turbo::c_push_heap(heap, greater());
        REQUIRE_EQ(0, heap[0]);
        turbo::c_pop_heap(heap, greater());
        REQUIRE_EQ(0, heap[3]);
        turbo::c_make_heap(heap, greater());
        turbo::c_sort_heap(heap, greater());
        REQUIRE_EQ(heap, std::vector<int>{3, 2, 1, 0});
        REQUIRE_FALSE(turbo::c_is_heap(heap, greater()));
    }

    TEST_CASE("MutatingTest, PermutationOperations") {
        std::vector<int> initial = {1, 2, 3, 4};
        std::vector<int> permuted = initial;

        turbo::c_next_permutation(permuted);
        REQUIRE(turbo::c_is_permutation(initial, permuted));
        REQUIRE(turbo::c_is_permutation(initial, permuted, std::equal_to<int>()));

        std::vector<int> permuted2 = initial;
        turbo::c_prev_permutation(permuted2, std::greater<int>());
        REQUIRE_EQ(permuted, permuted2);

        turbo::c_prev_permutation(permuted);
        REQUIRE_EQ(initial, permuted);
    }

}  // namespace
