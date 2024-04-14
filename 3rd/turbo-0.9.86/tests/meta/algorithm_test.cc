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

#include "turbo/meta/algorithm.h"

#include <algorithm>
#include <list>
#include <vector>

#include "turbo/platform/port.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

namespace {

    TEST_CASE("EqualTest, DefaultComparisonRandomAccess") {
        std::vector<int> v1{1, 2, 3};
        std::vector<int> v2 = v1;
        std::vector<int> v3 = {1, 2};
        std::vector<int> v4 = {1, 2, 4};

        CHECK(turbo::equal(v1.begin(), v1.end(), v2.begin(), v2.end()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v3.begin(), v3.end()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v4.begin(), v4.end()));
    }

    TEST_CASE("EqualTest, DefaultComparison") {
        std::list<int> lst1{1, 2, 3};
        std::list<int> lst2 = lst1;
        std::list<int> lst3{1, 2};
        std::list<int> lst4{1, 2, 4};

        CHECK(turbo::equal(lst1.begin(), lst1.end(), lst2.begin(), lst2.end()));
        CHECK_FALSE(turbo::equal(lst1.begin(), lst1.end(), lst3.begin(), lst3.end()));
        CHECK_FALSE(turbo::equal(lst1.begin(), lst1.end(), lst4.begin(), lst4.end()));
    }

    TEST_CASE("EqualTest, EmptyRange") {
        std::vector<int> v1{1, 2, 3};
        std::vector<int> empty1;
        std::vector<int> empty2;

        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105705
#if TURBO_HAVE_MIN_GNUC_VERSION(12, 0)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
#endif
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), empty1.begin(), empty1.end()));
#if TURBO_HAVE_MIN_GNUC_VERSION(12, 0)
#pragma GCC diagnostic pop
#endif
        CHECK_FALSE(turbo::equal(empty1.begin(), empty1.end(), v1.begin(), v1.end()));
        CHECK(
                turbo::equal(empty1.begin(), empty1.end(), empty2.begin(), empty2.end()));
    }

    TEST_CASE("EqualTest, MixedIterTypes") {
        std::vector<int> v1{1, 2, 3};
        std::list<int> lst1{v1.begin(), v1.end()};
        std::list<int> lst2{1, 2, 4};
        std::list<int> lst3{1, 2};

        CHECK(turbo::equal(v1.begin(), v1.end(), lst1.begin(), lst1.end()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), lst2.begin(), lst2.end()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), lst3.begin(), lst3.end()));
    }

    TEST_CASE("EqualTest, MixedValueTypes") {
        std::vector<int> v1{1, 2, 3};
        std::vector<char> v2{1, 2, 3};
        std::vector<char> v3{1, 2};
        std::vector<char> v4{1, 2, 4};

        CHECK(turbo::equal(v1.begin(), v1.end(), v2.begin(), v2.end()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v3.begin(), v3.end()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v4.begin(), v4.end()));
    }

    TEST_CASE("EqualTest, WeirdIterators") {
        std::vector<bool> v1{true, false};
        std::vector<bool> v2 = v1;
        std::vector<bool> v3{true};
        std::vector<bool> v4{true, true, true};

        CHECK(turbo::equal(v1.begin(), v1.end(), v2.begin(), v2.end()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v3.begin(), v3.end()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v4.begin(), v4.end()));
    }

    TEST_CASE("EqualTest, CustomComparison") {
        int n[] = {1, 2, 3, 4};
        std::vector<int *> v1{&n[0], &n[1], &n[2]};
        std::vector<int *> v2 = v1;
        std::vector<int *> v3{&n[0], &n[1], &n[3]};
        std::vector<int *> v4{&n[0], &n[1]};

        auto eq = [](int *a, int *b) { return *a == *b; };

        CHECK(turbo::equal(v1.begin(), v1.end(), v2.begin(), v2.end(), eq));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v3.begin(), v3.end(), eq));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v4.begin(), v4.end(), eq));
    }

    TEST_CASE("EqualTest, MoveOnlyPredicate") {
        std::vector<int> v1{1, 2, 3};
        std::vector<int> v2{4, 5, 6};

        // move-only equality predicate
        struct Eq {
            Eq() = default;

            Eq(Eq &&) = default;

            Eq(const Eq &) = delete;

            Eq &operator=(const Eq &) = delete;

            bool operator()(const int a, const int b) const { return a == b; }
        };

        CHECK(turbo::equal(v1.begin(), v1.end(), v1.begin(), v1.end(), Eq()));
        CHECK_FALSE(turbo::equal(v1.begin(), v1.end(), v2.begin(), v2.end(), Eq()));
    }

    struct CountingTrivialPred {
        int *count;

        bool operator()(int, int) const {
            ++*count;
            return true;
        }
    };

    TEST_CASE("EqualTest, RandomAccessComplexity") {
        std::vector<int> v1{1, 1, 3};
        std::vector<int> v2 = v1;
        std::vector<int> v3{1, 2};

        do {
            int count = 0;
            turbo::equal(v1.begin(), v1.end(), v2.begin(), v2.end(),
                         CountingTrivialPred{&count});
            CHECK_LE(count, 3);
        } while (std::next_permutation(v2.begin(), v2.end()));

        int count = 0;
        turbo::equal(v1.begin(), v1.end(), v3.begin(), v3.end(),
                     CountingTrivialPred{&count});
        CHECK_EQ(count, 0);
    }

    class LinearSearchTest {
    protected:
        LinearSearchTest() : container_{1, 2, 3} {}

        static bool Is3(int n) { return n == 3; }

        static bool Is4(int n) { return n == 4; }

        std::vector<int> container_;
    };

    TEST_CASE_FIXTURE(LinearSearchTest, "linear_search") {
        CHECK(turbo::linear_search(container_.begin(), container_.end(), 3));
        CHECK_FALSE(turbo::linear_search(container_.begin(), container_.end(), 4));
    }

    TEST_CASE_FIXTURE(LinearSearchTest, "linear_searchConst") {
        const std::vector<int> *const const_container = &container_;
        CHECK(
                turbo::linear_search(const_container->begin(), const_container->end(), 3));
        CHECK_FALSE(
                turbo::linear_search(const_container->begin(), const_container->end(), 4));
    }

    TEST_CASE("RotateTest, Rotate") {
        std::vector<int> v{0, 1, 2, 3, 4};
        CHECK_EQ(*turbo::rotate(v.begin(), v.begin() + 2, v.end()), 0);
        CHECK_EQ(v, std::vector<int>{2, 3, 4, 0, 1});

        std::list<int> l{0, 1, 2, 3, 4};
        CHECK_EQ(*turbo::rotate(l.begin(), std::next(l.begin(), 3), l.end()), 0);
        CHECK_EQ(l, std::list<int>{3, 4, 0, 1, 2});
    }

    // --------------------------------------------------------
    // Testcase: distance
    // --------------------------------------------------------
    TEST_CASE("distance.integral" * doctest::timeout(300)) {

        auto count = [](int beg, int end, int step) {
            size_t c = 0;
            for (int i = beg; step > 0 ? i < end : i > end; i += step) {
                ++c;
            }
            return c;
        };

        for (int beg = -50; beg <= 50; ++beg) {
            for (int end = -50; end <= 50; ++end) {
                if (beg < end) {   // positive step
                    for (int s = 1; s <= 50; s++) {
                        REQUIRE((turbo::distance(beg, end, s) == count(beg, end, s)));
                    }
                } else {            // negative step
                    for (int s = -1; s >= -50; s--) {
                        REQUIRE((turbo::distance(beg, end, s) == count(beg, end, s)));
                    }
                }
            }
        }

    }

}  // namespace
