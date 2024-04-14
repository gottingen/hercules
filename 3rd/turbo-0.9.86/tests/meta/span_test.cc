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

#include "turbo/meta/span.h"

#include <array>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "turbo/container/fixed_array.h"
#include "turbo/container/inlined_vector.h"
#include "turbo/platform/port.h"
#include "turbo/platform/options.h"
#include "turbo/format/format.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

namespace {
    
    std::vector<int> MakeRamp(int len, int offset = 0) {
        std::vector<int> v(len);
        std::iota(v.begin(), v.end(), offset);
        return v;
    }

    TEST_CASE("IntSpan, EmptyCtors") {
        turbo::Span<int> s;
        REQUIRE(s.empty());
    }

    TEST_CASE("IntSpan, PtrLenCtor") {
        int a[] = {1, 2, 3};
        turbo::Span<int> s(&a[0], 2);
        REQUIRE_EQ(s.size(), 2);
    }

    TEST_CASE("IntSpan, ArrayCtor") {
        int a[] = {1, 2, 3};
        turbo::Span<int> s(a);
        REQUIRE_EQ(s.size(), 3);

        REQUIRE((std::is_constructible<turbo::Span<const int>, int[3]>::value));
        REQUIRE(
                (std::is_constructible<turbo::Span<const int>, const int[3]>::value));
        REQUIRE_FALSE((std::is_constructible<turbo::Span<int>, const int[3]>::value));
        REQUIRE((std::is_convertible<int[3], turbo::Span<const int>>::value));
        REQUIRE(
                (std::is_convertible<const int[3], turbo::Span<const int>>::value));
    }

    template<typename T>
    void TakesGenericSpan(turbo::Span<T>) {}

    TEST_CASE("IntSpan, ContainerCtor") {
        std::vector<int> empty;
        std::vector<int> filled{1, 2, 3};

        REQUIRE(
                (std::is_convertible<std::vector<int> &, turbo::Span<const int>>::value));
        REQUIRE(
                (std::is_convertible<turbo::Span<int> &, turbo::Span<const int>>::value));

        TakesGenericSpan(turbo::Span<int>(filled));
    }

// A struct supplying shallow data() const.
    struct ContainerWithShallowConstData {
        std::vector<int> storage;

        int *data() const { return const_cast<int *>(storage.data()); }

        int size() const { return storage.size(); }
    };

    TEST_CASE("IntSpan, ShallowConstness") {
        const ContainerWithShallowConstData c{MakeRamp(20)};
        turbo::Span<int> s(
                c);  // We should be able to do this even though data() is const.
        s[0] = -1;
        REQUIRE_EQ(c.storage[0], -1);
    }

    TEST_CASE("CharSpan, StringCtor") {
        std::string empty = "";
        REQUIRE_FALSE((std::is_constructible<turbo::Span<int>, std::string>::value));
        REQUIRE_FALSE(
                (std::is_constructible<turbo::Span<const int>, std::string>::value));
        REQUIRE(
                (std::is_convertible<std::string, turbo::Span<const char>>::value));
    }

    TEST_CASE("IntSpan, FromConstPointer") {
        REQUIRE((std::is_constructible<turbo::Span<const int *const>,
                std::vector<int *>>::value));
        REQUIRE((std::is_constructible<turbo::Span<const int *const>,
                std::vector<const int *>>::value));
        REQUIRE_FALSE((
                             std::is_constructible<turbo::Span<const int *>, std::vector<int *>>::value));
        REQUIRE_FALSE((
                             std::is_constructible<turbo::Span<int *>, std::vector<const int *>>::value));
    }

    struct TypeWithMisleadingData {
        int &data() { return i; }

        int size() { return 1; }

        int i;
    };

    struct TypeWithMisleadingSize {
        int *data() { return &i; }

        const char *size() { return "1"; }

        int i;
    };

    TEST_CASE("IntSpan, EvilTypes") {
        REQUIRE_FALSE(
                (std::is_constructible<turbo::Span<int>, TypeWithMisleadingData &>::value));
        REQUIRE_FALSE(
                (std::is_constructible<turbo::Span<int>, TypeWithMisleadingSize &>::value));
    }

    struct Base {
        int *data() { return &i; }

        int size() { return 1; }

        int i;
    };

    struct Derived : Base {
    };

    TEST_CASE("IntSpan, SpanOfDerived") {
        REQUIRE((std::is_constructible<turbo::Span<int>, Base &>::value));
        REQUIRE((std::is_constructible<turbo::Span<int>, Derived &>::value));
        REQUIRE_FALSE(
                (std::is_constructible<turbo::Span<Base>, std::vector<Derived>>::value));
    }

    void TestInitializerList(turbo::Span<const int> s, const std::vector<int> &v) {
        REQUIRE(turbo::equal(s.begin(), s.end(), v.begin(), v.end()));
    }

    TEST_CASE("ConstIntSpan, InitializerListConversion") {
        TestInitializerList({}, {});
        TestInitializerList({1}, {1});
        TestInitializerList({1, 2, 3}, {1, 2, 3});

        REQUIRE_FALSE((std::is_constructible<turbo::Span<int>,
                std::initializer_list<int>>::value));
        REQUIRE_FALSE((
                             std::is_convertible<turbo::Span<int>, std::initializer_list<int>>::value));
    }

    TEST_CASE("IntSpan, Data") {
        int i;
        turbo::Span<int> s(&i, 1);
        REQUIRE_EQ(&i, s.data());
    }

    TEST_CASE("IntSpan, SizeLengthEmpty") {
        turbo::Span<int> empty;
        REQUIRE_EQ(empty.size(), 0);
        REQUIRE(empty.empty());
        REQUIRE_EQ(empty.size(), empty.length());

        auto v = MakeRamp(10);
        turbo::Span<int> s(v);
        REQUIRE_EQ(s.size(), 10);
        REQUIRE_FALSE(s.empty());
        REQUIRE_EQ(s.size(), s.length());
    }

    TEST_CASE("IntSpan, ElementAccess") {
        auto v = MakeRamp(10);
        turbo::Span<int> s(v);
        for (int i = 0; i < s.size(); ++i) {
            REQUIRE_EQ(s[i], s.at(i));
        }

        REQUIRE_EQ(s.front(), s[0]);
        REQUIRE_EQ(s.back(), s[9]);
    }

    TEST_CASE("IntSpan, AtThrows") {
        auto v = MakeRamp(10);
        turbo::Span<int> s(v);

        REQUIRE_EQ(s.at(9), 9);
        REQUIRE_THROWS_AS(s.at(10), std::out_of_range);
    }

    TEST_CASE("IntSpan, RemovePrefixAndSuffix") {
        auto v = MakeRamp(20, 1);
        turbo::Span<int> s(v);
        REQUIRE_EQ(s.size(), 20);

        s.remove_suffix(0);
        s.remove_prefix(0);
        REQUIRE_EQ(s.size(), 20);

        s.remove_prefix(1);
        REQUIRE_EQ(s.size(), 19);
        REQUIRE_EQ(s[0], 2);

        s.remove_suffix(1);
        REQUIRE_EQ(s.size(), 18);
        REQUIRE_EQ(s.back(), 19);

        s.remove_prefix(7);
        REQUIRE_EQ(s.size(), 11);
        REQUIRE_EQ(s[0], 9);

        s.remove_suffix(11);
        REQUIRE_EQ(s.size(), 0);

        REQUIRE_EQ(v, MakeRamp(20, 1));
    }

    TEST_CASE("IntSpan, Equality") {
        const int arr1[] = {1, 2, 3, 4, 5};
        int arr2[] = {1, 2, 3, 4, 5};
        std::vector<int> vec1(std::begin(arr1), std::end(arr1));
        std::vector<int> vec2 = vec1;
        std::vector<int> other_vec = {2, 4, 6, 8, 10};
        // These two slices are from different vectors, but have the same size and
        // have the same elements (right now).  They should compare equal. Test both
        // == and !=.
        const turbo::Span<const int> from1 = vec1;
        const turbo::Span<const int> from2 = vec2;
        REQUIRE_EQ(from1, from1);
        REQUIRE_FALSE(from1 != from1);
        REQUIRE_EQ(from1, from2);
        REQUIRE_FALSE(from1 != from2);

        // These two slices have different underlying vector values. They should be
        // considered not equal. Test both == and !=.
        const turbo::Span<const int> from_other = other_vec;
        REQUIRE_NE(from1, from_other);
        REQUIRE_FALSE(from1 == from_other);

        // Comparison between a vector and its slice should be equal. And vice-versa.
        // This ensures implicit conversion to Span works on both sides of ==.
        REQUIRE_EQ(vec1, from1);
        REQUIRE_FALSE(vec1 != from1);
        REQUIRE_EQ(from1, vec1);
        REQUIRE_FALSE(from1 != vec1);

        // This verifies that turbo::Span<T> can be compared freely with
        // turbo::Span<const T>.
        const turbo::Span<int> mutable_from1(vec1);
        const turbo::Span<int> mutable_from2(vec2);
        REQUIRE_EQ(from1, mutable_from1);
        REQUIRE_EQ(mutable_from1, from1);
        REQUIRE_EQ(mutable_from1, mutable_from2);
        REQUIRE_EQ(mutable_from2, mutable_from1);

        // Comparison between a vector and its slice should be equal for mutable
        // Spans as well.
        REQUIRE_EQ(vec1, mutable_from1);
        REQUIRE_FALSE(vec1 != mutable_from1);
        REQUIRE_EQ(mutable_from1, vec1);
        REQUIRE_FALSE(mutable_from1 != vec1);

        // Comparison between convertible-to-Span-of-const and Span-of-mutable. Arrays
        // are used because they're the only value type which converts to a
        // Span-of-mutable. REQUIRE is used instead of REQUIRE_EQ to avoid
        // array-to-pointer decay.
        REQUIRE(arr1 == mutable_from1);
        REQUIRE_FALSE(arr1 != mutable_from1);
        REQUIRE(mutable_from1 == arr1);
        REQUIRE_FALSE(mutable_from1 != arr1);

        // Comparison between convertible-to-Span-of-mutable and Span-of-const
        REQUIRE(arr2 == from1);
        REQUIRE_FALSE(arr2 != from1);
        REQUIRE(from1 == arr2);
        REQUIRE_FALSE(from1 != arr2);

        // With a different size, the array slices should not be equal.
        REQUIRE_NE(from1, turbo::Span<const int>(from1).subspan(0, from1.size() - 1));

        // With different contents, the array slices should not be equal.
        ++vec2.back();
        REQUIRE_NE(from1, from2);
    }

    class IntSpanOrderComparisonTest{
    public:
        IntSpanOrderComparisonTest()
                : arr_before_{1, 2, 3},
                  arr_after_{1, 2, 4},
                  carr_after_{1, 2, 4},
                  vec_before_(std::begin(arr_before_), std::end(arr_before_)),
                  vec_after_(std::begin(arr_after_), std::end(arr_after_)),
                  before_(vec_before_),
                  after_(vec_after_),
                  cbefore_(vec_before_),
                  cafter_(vec_after_) {}

    protected:
        int arr_before_[3], arr_after_[3];
        const int carr_after_[3];
        std::vector<int> vec_before_, vec_after_;
        turbo::Span<int> before_, after_;
        turbo::Span<const int> cbefore_, cafter_;
    };

    TEST_CASE_FIXTURE(IntSpanOrderComparisonTest," CompareSpans") {
        REQUIRE(cbefore_ < cafter_);
        REQUIRE(cbefore_ <= cafter_);
        REQUIRE(cafter_ > cbefore_);
        REQUIRE(cafter_ >= cbefore_);

        REQUIRE_FALSE(cbefore_ > cafter_);
        REQUIRE_FALSE(cafter_ < cbefore_);

        REQUIRE(before_ < after_);
        REQUIRE(before_ <= after_);
        REQUIRE(after_ > before_);
        REQUIRE(after_ >= before_);

        REQUIRE_FALSE(before_ > after_);
        REQUIRE_FALSE(after_ < before_);

        REQUIRE(cbefore_ < after_);
        REQUIRE(cbefore_ <= after_);
        REQUIRE(after_ > cbefore_);
        REQUIRE(after_ >= cbefore_);

        REQUIRE_FALSE(cbefore_ > after_);
        REQUIRE_FALSE(after_ < cbefore_);
    }

    TEST_CASE_FIXTURE(IntSpanOrderComparisonTest," SpanOfConstAndContainer") {
        REQUIRE(cbefore_ < vec_after_);
        REQUIRE(cbefore_ <= vec_after_);
        REQUIRE(vec_after_ > cbefore_);
        REQUIRE(vec_after_ >= cbefore_);

        REQUIRE_FALSE(cbefore_ > vec_after_);
        REQUIRE_FALSE(vec_after_ < cbefore_);

        REQUIRE(arr_before_ < cafter_);
        REQUIRE(arr_before_ <= cafter_);
        REQUIRE(cafter_ > arr_before_);
        REQUIRE(cafter_ >= arr_before_);

        REQUIRE_FALSE(arr_before_ > cafter_);
        REQUIRE_FALSE(cafter_ < arr_before_);
    }

    TEST_CASE_FIXTURE(IntSpanOrderComparisonTest," SpanOfMutableAndContainer") {
        REQUIRE(vec_before_ < after_);
        REQUIRE(vec_before_ <= after_);
        REQUIRE(after_ > vec_before_);
        REQUIRE(after_ >= vec_before_);

        REQUIRE_FALSE(vec_before_ > after_);
        REQUIRE_FALSE(after_ < vec_before_);

        REQUIRE(before_ < carr_after_);
        REQUIRE(before_ <= carr_after_);
        REQUIRE(carr_after_ > before_);
        REQUIRE(carr_after_ >= before_);

        REQUIRE_FALSE(before_ > carr_after_);
        REQUIRE_FALSE(carr_after_ < before_);
    }

    TEST_CASE_FIXTURE(IntSpanOrderComparisonTest," EqualSpans") {
        REQUIRE_FALSE(before_ < before_);
        REQUIRE(before_ <= before_);
        REQUIRE_FALSE(before_ > before_);
        REQUIRE(before_ >= before_);
    }

    TEST_CASE_FIXTURE(IntSpanOrderComparisonTest," Subspans") {
        auto subspan = before_.subspan(0, 1);
        REQUIRE(subspan < before_);
        REQUIRE(subspan <= before_);
        REQUIRE(before_ > subspan);
        REQUIRE(before_ >= subspan);

        REQUIRE_FALSE(subspan > before_);
        REQUIRE_FALSE(before_ < subspan);
    }

    TEST_CASE_FIXTURE(IntSpanOrderComparisonTest," EmptySpans") {
        turbo::Span<int> empty;
        REQUIRE_FALSE(empty < empty);
        REQUIRE(empty <= empty);
        REQUIRE_FALSE(empty > empty);
        REQUIRE(empty >= empty);

        REQUIRE(empty < before_);
        REQUIRE(empty <= before_);
        REQUIRE(before_ > empty);
        REQUIRE(before_ >= empty);

        REQUIRE_FALSE(empty > before_);
        REQUIRE_FALSE(before_ < empty);
    }

    TEST_CASE("IntSpan, IteratorsAndReferences") {
        auto accept_pointer = [](int *) {};
        auto accept_reference = [](int &) {};
        auto accept_iterator = [](turbo::Span<int>::iterator) {};
        auto accept_const_iterator = [](turbo::Span<int>::const_iterator) {};
        auto accept_reverse_iterator = [](turbo::Span<int>::reverse_iterator) {};
        auto accept_const_reverse_iterator =
                [](turbo::Span<int>::const_reverse_iterator) {};

        int a[1];
        turbo::Span<int> s = a;

        accept_pointer(s.data());
        accept_iterator(s.begin());
        accept_const_iterator(s.begin());
        accept_const_iterator(s.cbegin());
        accept_iterator(s.end());
        accept_const_iterator(s.end());
        accept_const_iterator(s.cend());
        accept_reverse_iterator(s.rbegin());
        accept_const_reverse_iterator(s.rbegin());
        accept_const_reverse_iterator(s.crbegin());
        accept_reverse_iterator(s.rend());
        accept_const_reverse_iterator(s.rend());
        accept_const_reverse_iterator(s.crend());

        accept_reference(s[0]);
        accept_reference(s.at(0));
        accept_reference(s.front());
        accept_reference(s.back());
    }

    TEST_CASE("IntSpan, IteratorsAndReferences_Const") {
        auto accept_pointer = [](int *) {};
        auto accept_reference = [](int &) {};
        auto accept_iterator = [](turbo::Span<int>::iterator) {};
        auto accept_const_iterator = [](turbo::Span<int>::const_iterator) {};
        auto accept_reverse_iterator = [](turbo::Span<int>::reverse_iterator) {};
        auto accept_const_reverse_iterator =
                [](turbo::Span<int>::const_reverse_iterator) {};

        int a[1];
        const turbo::Span<int> s = a;

        accept_pointer(s.data());
        accept_iterator(s.begin());
        accept_const_iterator(s.begin());
        accept_const_iterator(s.cbegin());
        accept_iterator(s.end());
        accept_const_iterator(s.end());
        accept_const_iterator(s.cend());
        accept_reverse_iterator(s.rbegin());
        accept_const_reverse_iterator(s.rbegin());
        accept_const_reverse_iterator(s.crbegin());
        accept_reverse_iterator(s.rend());
        accept_const_reverse_iterator(s.rend());
        accept_const_reverse_iterator(s.crend());

        accept_reference(s[0]);
        accept_reference(s.at(0));
        accept_reference(s.front());
        accept_reference(s.back());
    }

    TEST_CASE("IntSpan, NoexceptTest") {
        int a[] = {1, 2, 3};
        std::vector<int> v;
        REQUIRE(noexcept(turbo::Span<const int>()));
        REQUIRE(noexcept(turbo::Span<const int>(a, 2)));
        REQUIRE(noexcept(turbo::Span<const int>(a)));
        REQUIRE(noexcept(turbo::Span<const int>(v)));
        REQUIRE(noexcept(turbo::Span<int>(v)));
        REQUIRE(noexcept(turbo::Span<const int>({1, 2, 3})));
        REQUIRE(noexcept(turbo::MakeSpan(v)));
        REQUIRE(noexcept(turbo::MakeSpan(a)));
        REQUIRE(noexcept(turbo::MakeSpan(a, 2)));
        REQUIRE(noexcept(turbo::MakeSpan(a, a + 1)));
        REQUIRE(noexcept(turbo::MakeConstSpan(v)));
        REQUIRE(noexcept(turbo::MakeConstSpan(a)));
        REQUIRE(noexcept(turbo::MakeConstSpan(a, 2)));
        REQUIRE(noexcept(turbo::MakeConstSpan(a, a + 1)));

        turbo::Span<int> s(v);
        REQUIRE(noexcept(s.data()));
        REQUIRE(noexcept(s.size()));
        REQUIRE(noexcept(s.length()));
        REQUIRE(noexcept(s.empty()));
        REQUIRE(noexcept(s[0]));
        REQUIRE(noexcept(s.front()));
        REQUIRE(noexcept(s.back()));
        REQUIRE(noexcept(s.begin()));
        REQUIRE(noexcept(s.cbegin()));
        REQUIRE(noexcept(s.end()));
        REQUIRE(noexcept(s.cend()));
        REQUIRE(noexcept(s.rbegin()));
        REQUIRE(noexcept(s.crbegin()));
        REQUIRE(noexcept(s.rend()));
        REQUIRE(noexcept(s.crend()));
        REQUIRE(noexcept(s.remove_prefix(0)));
        REQUIRE(noexcept(s.remove_suffix(0)));
    }

// ConstexprTester exercises expressions in a constexpr context. Simply placing
// the expression in a constexpr function is not enough, as some compilers will
// simply compile the constexpr function as runtime code. Using template
// parameters forces compile-time execution.
    template<int i>
    struct ConstexprTester {
    };

#define TURBO_TEST_CONSTEXPR(expr)                       \
  do {                                                  \
    TURBO_MAYBE_UNUSED ConstexprTester<(expr, 1)> t; \
  } while (0)

    struct ContainerWithConstexprMethods {
        constexpr int size() const { return 1; }

        constexpr const int *data() const { return &i; }

        const int i;
    };

    TEST_CASE("ConstIntSpan, ConstexprTest") {
        static constexpr int a[] = {1, 2, 3};
        static constexpr int sized_arr[2] = {1, 2};
        static constexpr ContainerWithConstexprMethods c{1};
        TURBO_TEST_CONSTEXPR(turbo::Span<const int>());
        TURBO_TEST_CONSTEXPR(turbo::Span<const int>(a, 2));
        TURBO_TEST_CONSTEXPR(turbo::Span<const int>(sized_arr));
        TURBO_TEST_CONSTEXPR(turbo::Span<const int>(c));
        TURBO_TEST_CONSTEXPR(turbo::MakeSpan(&a[0], 1));
        TURBO_TEST_CONSTEXPR(turbo::MakeSpan(c));
        TURBO_TEST_CONSTEXPR(turbo::MakeSpan(a));
        TURBO_TEST_CONSTEXPR(turbo::MakeConstSpan(&a[0], 1));
        TURBO_TEST_CONSTEXPR(turbo::MakeConstSpan(c));
        TURBO_TEST_CONSTEXPR(turbo::MakeConstSpan(a));

        constexpr turbo::Span<const int> span = c;
        TURBO_TEST_CONSTEXPR(span.data());
        TURBO_TEST_CONSTEXPR(span.size());
        TURBO_TEST_CONSTEXPR(span.length());
        TURBO_TEST_CONSTEXPR(span.empty());
        TURBO_TEST_CONSTEXPR(span.begin());
        TURBO_TEST_CONSTEXPR(span.cbegin());
        TURBO_TEST_CONSTEXPR(span.subspan(0, 0));
        TURBO_TEST_CONSTEXPR(span.first(1));
        TURBO_TEST_CONSTEXPR(span.last(1));
        TURBO_TEST_CONSTEXPR(span[0]);
    }

    struct BigStruct {
        char bytes[10000];
    };

    TEST_CASE("Span, SpanSize") {
        REQUIRE_LE(sizeof(turbo::Span<int>), 2 * sizeof(void *));
        REQUIRE_LE(sizeof(turbo::Span<BigStruct>), 2 * sizeof(void *));
    }

}  // namespace
