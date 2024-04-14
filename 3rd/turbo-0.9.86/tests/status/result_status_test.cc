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

#include "turbo/status/result_status.h"

#include <array>
#include <initializer_list>
#include <memory>
#include <string>
#include <any>
#include <type_traits>
#include <utility>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "turbo/base/casts.h"
#include "turbo/memory/memory.h"
#include "turbo/status/status.h"
#include "turbo/strings/string_view.h"
#include "turbo/meta/utility.h"
#include <variant>

namespace {

    using ::testing::AllOf;
    using ::testing::AnyWith;
    using ::testing::ElementsAre;
    using ::testing::Field;
    using ::testing::HasSubstr;
    using ::testing::Ne;
    using ::testing::Not;
    using ::testing::Pointee;
    using ::testing::VariantWith;

#ifdef GTEST_HAS_STATUS_MATCHERS
    using ::testing::status::IsOk;
    using ::testing::status::IsOkAndHolds;
#else  // GTEST_HAS_STATUS_MATCHERS

    inline const ::turbo::Status &GetStatus(const ::turbo::Status &status) {
        return status;
    }

    template<typename T>
    inline const ::turbo::Status &GetStatus(const ::turbo::ResultStatus<T> &status) {
        return status.status();
    }

// Monomorphic implementation of matcher IsOkAndHolds(m).  ResultStatusType is a
// reference to ResultStatus<T>.
    template<typename ResultStatusType>
    class IsOkAndHoldsMatcherImpl
            : public ::testing::MatcherInterface<ResultStatusType> {
    public:
        typedef
        typename std::remove_reference<ResultStatusType>::type::value_type value_type;

        template<typename InnerMatcher>
        explicit IsOkAndHoldsMatcherImpl(InnerMatcher &&inner_matcher)
                : inner_matcher_(::testing::SafeMatcherCast<const value_type &>(
                std::forward<InnerMatcher>(inner_matcher))) {}

        void DescribeTo(std::ostream *os) const override {
            *os << "is OK and has a value that ";
            inner_matcher_.DescribeTo(os);
        }

        void DescribeNegationTo(std::ostream *os) const override {
            *os << "isn't OK or has a value that ";
            inner_matcher_.DescribeNegationTo(os);
        }

        bool MatchAndExplain(
                ResultStatusType actual_value,
                ::testing::MatchResultListener *result_listener) const override {
            if (!actual_value.ok()) {
                *result_listener << "which has status " << actual_value.status();
                return false;
            }

            ::testing::StringMatchResultListener inner_listener;
            const bool matches =
                    inner_matcher_.MatchAndExplain(*actual_value, &inner_listener);
            const std::string inner_explanation = inner_listener.str();
            if (!inner_explanation.empty()) {
                *result_listener << "which contains value "
                                 << ::testing::PrintToString(*actual_value) << ", "
                                 << inner_explanation;
            }
            return matches;
        }

    private:
        const ::testing::Matcher<const value_type &> inner_matcher_;
    };

// Implements IsOkAndHolds(m) as a polymorphic matcher.
    template<typename InnerMatcher>
    class IsOkAndHoldsMatcher {
    public:
        explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
                : inner_matcher_(std::move(inner_matcher)) {}

        // Converts this polymorphic matcher to a monomorphic matcher of the
        // given type.  ResultStatusType can be either ResultStatus<T> or a
        // reference to ResultStatus<T>.
        template<typename ResultStatusType>
        operator ::testing::Matcher<ResultStatusType>() const {  // NOLINT
            return ::testing::Matcher<ResultStatusType>(
                    new IsOkAndHoldsMatcherImpl<const ResultStatusType &>(inner_matcher_));
        }

    private:
        const InnerMatcher inner_matcher_;
    };

// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Status, ResultStatus<>, or a reference to either of them.
    template<typename T>
    class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
    public:
        void DescribeTo(std::ostream *os) const override { *os << "is OK"; }

        void DescribeNegationTo(std::ostream *os) const override {
            *os << "is not OK";
        }

        bool MatchAndExplain(T actual_value,
                             ::testing::MatchResultListener *) const override {
            return GetStatus(actual_value).ok();
        }
    };

// Implements IsOk() as a polymorphic matcher.
    class IsOkMatcher {
    public:
        template<typename T>
        operator ::testing::Matcher<T>() const {  // NOLINT
            return ::testing::Matcher<T>(new MonoIsOkMatcherImpl<T>());
        }
    };

// Macros for testing the results of functions that return turbo::Status or
// turbo::ResultStatus<T> (for any type T).
#define EXPECT_OK(expression) EXPECT_THAT(expression, IsOk())

// Returns a gMock matcher that matches a ResultStatus<> whose status is
// OK and whose value matches the inner matcher.
    template<typename InnerMatcher>
    IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type> IsOkAndHolds(
            InnerMatcher &&inner_matcher) {
        return IsOkAndHoldsMatcher<typename std::decay<InnerMatcher>::type>(
                std::forward<InnerMatcher>(inner_matcher));
    }

// Returns a gMock matcher that matches a Status or ResultStatus<> which is OK.
    inline IsOkMatcher IsOk() { return IsOkMatcher(); }

#endif  // GTEST_HAS_STATUS_MATCHERS

    struct CopyDetector {
        CopyDetector() = default;

        explicit CopyDetector(int xx) : x(xx) {}

        CopyDetector(CopyDetector &&d) noexcept
                : x(d.x), copied(false), moved(true) {}

        CopyDetector(const CopyDetector &d) : x(d.x), copied(true), moved(false) {}

        CopyDetector &operator=(const CopyDetector &c) {
            x = c.x;
            copied = true;
            moved = false;
            return *this;
        }

        CopyDetector &operator=(CopyDetector &&c) noexcept {
            x = c.x;
            copied = false;
            moved = true;
            return *this;
        }

        int x = 0;
        bool copied = false;
        bool moved = false;
    };

    testing::Matcher<const CopyDetector &> CopyDetectorHas(int a, bool b, bool c) {
        return AllOf(Field(&CopyDetector::x, a), Field(&CopyDetector::moved, b),
                     Field(&CopyDetector::copied, c));
    }

    class Base1 {
    public:
        virtual ~Base1() {}

        int pad;
    };

    class Base2 {
    public:
        virtual ~Base2() {}

        int yetotherpad;
    };

    class Derived : public Base1, public Base2 {
    public:
        virtual ~Derived() {}

        int evenmorepad;
    };

    class CopyNoAssign {
    public:
        explicit CopyNoAssign(int value) : foo(value) {}

        CopyNoAssign(const CopyNoAssign &other) : foo(other.foo) {}

        int foo;

    private:
        const CopyNoAssign &operator=(const CopyNoAssign &);
    };

    turbo::ResultStatus<std::unique_ptr<int>> ReturnUniquePtr() {
        // Uses implicit constructor from T&&
        return std::make_unique<int>(0);
    }

    TEST(ResultStatus, ElementType) {
        static_assert(std::is_same<turbo::ResultStatus<int>::value_type, int>(), "");
        static_assert(std::is_same<turbo::ResultStatus<char>::value_type, char>(), "");
    }

    TEST(ResultStatus, TestMoveOnlyInitialization) {
        turbo::ResultStatus<std::unique_ptr<int>> thing(ReturnUniquePtr());
        ASSERT_TRUE(thing.ok());
        EXPECT_EQ(0, **thing);
        int *previous = thing->get();

        thing = ReturnUniquePtr();
        EXPECT_TRUE(thing.ok());
        EXPECT_EQ(0, **thing);
        EXPECT_NE(previous, thing->get());
    }

    TEST(ResultStatus, TestMoveOnlyValueExtraction) {
        turbo::ResultStatus<std::unique_ptr<int>> thing(ReturnUniquePtr());
        ASSERT_TRUE(thing.ok());
        std::unique_ptr<int> ptr = *std::move(thing);
        EXPECT_EQ(0, *ptr);

        thing = std::move(ptr);
        ptr = std::move(*thing);
        EXPECT_EQ(0, *ptr);
    }

    TEST(ResultStatus, TestMoveOnlyInitializationFromTemporaryByValueOrDie) {
        std::unique_ptr<int> ptr(*ReturnUniquePtr());
        EXPECT_EQ(0, *ptr);
    }

    TEST(ResultStatus, TestValueOrDieOverloadForConstTemporary) {
        static_assert(
                std::is_same<
                        const int &&,
                        decltype(std::declval<const turbo::ResultStatus<int> &&>().value())>(),
                "value() for const temporaries should return const T&&");
    }

    TEST(ResultStatus, TestMoveOnlyConversion) {
        turbo::ResultStatus<std::unique_ptr<const int>> const_thing(ReturnUniquePtr());
        EXPECT_TRUE(const_thing.ok());
        EXPECT_EQ(0, **const_thing);

        // Test rvalue converting assignment
        const int *const_previous = const_thing->get();
        const_thing = ReturnUniquePtr();
        EXPECT_TRUE(const_thing.ok());
        EXPECT_EQ(0, **const_thing);
        EXPECT_NE(const_previous, const_thing->get());
    }

    TEST(ResultStatus, TestMoveOnlyVector) {
        // Sanity check that turbo::ResultStatus<MoveOnly> works in vector.
        std::vector<turbo::ResultStatus<std::unique_ptr<int>>> vec;
        vec.push_back(ReturnUniquePtr());
        vec.resize(2);
        auto another_vec = std::move(vec);
        EXPECT_EQ(0, **another_vec[0]);
        EXPECT_EQ(turbo::unknown_error(""), another_vec[1].status());
    }

    TEST(ResultStatus, TestDefaultCtor) {
        turbo::ResultStatus<int> thing;
        EXPECT_FALSE(thing.ok());
        EXPECT_EQ(thing.status().map_code(), turbo::kUnknown);
    }

    TEST(ResultStatus, StatusCtorForwards) {
        turbo::Status status(turbo::kInternal, "Some error");

        EXPECT_EQ(turbo::ResultStatus<int>(status).status().message(), "Some error");
        EXPECT_EQ(status.message(), "Some error");

        EXPECT_EQ(turbo::ResultStatus<int>(std::move(status)).status().message(),
                  "Some error");
        EXPECT_NE(status.message(), "Some error");
    }

    TEST(BadResultStatusAccessTest, CopyConstructionWhatOk) {
        turbo::Status error =
                turbo::internal_error("some arbitrary message too big for the sso buffer");
        turbo::BadResultStatusAccess e1{error};
        turbo::BadResultStatusAccess e2{e1};
        EXPECT_THAT(e1.what(), HasSubstr(error.to_string()));
        EXPECT_THAT(e2.what(), HasSubstr(error.to_string()));
    }

    TEST(BadResultStatusAccessTest, CopyAssignmentWhatOk) {
        turbo::Status error =
                turbo::internal_error("some arbitrary message too big for the sso buffer");
        turbo::BadResultStatusAccess e1{error};
        turbo::BadResultStatusAccess e2{turbo::internal_error("other")};
        e2 = e1;
        EXPECT_THAT(e1.what(), HasSubstr(error.to_string()));
        EXPECT_THAT(e2.what(), HasSubstr(error.to_string()));
    }

    TEST(BadResultStatusAccessTest, MoveConstructionWhatOk) {
        turbo::Status error =
                turbo::internal_error("some arbitrary message too big for the sso buffer");
        turbo::BadResultStatusAccess e1{error};
        turbo::BadResultStatusAccess e2{std::move(e1)};
        EXPECT_THAT(e2.what(), HasSubstr(error.to_string()));
    }

    TEST(BadResultStatusAccessTest, MoveAssignmentWhatOk) {
        turbo::Status error =
                turbo::internal_error("some arbitrary message too big for the sso buffer");
        turbo::BadResultStatusAccess e1{error};
        turbo::BadResultStatusAccess e2{turbo::internal_error("other")};
        e2 = std::move(e1);
        EXPECT_THAT(e2.what(), HasSubstr(error.to_string()));
    }

// Define `EXPECT_DEATH_OR_THROW` to test the behavior of `ResultStatus::value`,
// which either throws `BadResultStatusAccess` or `TURBO_LOG(FATAL)` based on whether
// exceptions are enabled.
#ifdef TURBO_HAVE_EXCEPTIONS
#define EXPECT_DEATH_OR_THROW(statement, status_)                  \
  EXPECT_THROW(                                                    \
      {                                                            \
        try {                                                      \
          statement;                                               \
        } catch (const turbo::BadResultStatusAccess& e) {               \
          EXPECT_EQ(e.status(), status_);                          \
          EXPECT_THAT(e.what(), HasSubstr(e.status().to_string())); \
          throw;                                                   \
        }                                                          \
      },                                                           \
      turbo::BadResultStatusAccess);
#else  // TURBO_HAVE_EXCEPTIONS
#define EXPECT_DEATH_OR_THROW(statement, status) \
  EXPECT_DEATH_IF_SUPPORTED(statement, status.to_string());
#endif  // TURBO_HAVE_EXCEPTIONS

    TEST(ResultStatusDeathTest, TestDefaultCtorValue) {
        turbo::ResultStatus<int> thing;
        EXPECT_DEATH_OR_THROW(thing.value(), turbo::unknown_error(""));
        const turbo::ResultStatus<int> thing2;
        EXPECT_DEATH_OR_THROW(thing2.value(), turbo::unknown_error(""));
    }

    TEST(ResultStatusDeathTest, TestValueNotOk) {
        turbo::ResultStatus<int> thing(turbo::cancelled_error());
        EXPECT_DEATH_OR_THROW(thing.value(), turbo::cancelled_error());
    }

    TEST(ResultStatusDeathTest, TestValueNotOkConst) {
        const turbo::ResultStatus<int> thing(turbo::unknown_error(""));
        EXPECT_DEATH_OR_THROW(thing.value(), turbo::unknown_error(""));
    }

    TEST(ResultStatusDeathTest, TestPointerDefaultCtorValue) {
        turbo::ResultStatus<int *> thing;
        EXPECT_DEATH_OR_THROW(thing.value(), turbo::unknown_error(""));
    }

    TEST(ResultStatusDeathTest, TestPointerValueNotOk) {
        turbo::ResultStatus<int *> thing(turbo::cancelled_error());
        EXPECT_DEATH_OR_THROW(thing.value(), turbo::cancelled_error());
    }

    TEST(ResultStatusDeathTest, TestPointerValueNotOkConst) {
        const turbo::ResultStatus<int *> thing(turbo::cancelled_error());
        EXPECT_DEATH_OR_THROW(thing.value(), turbo::cancelled_error());
    }

#if GTEST_HAS_DEATH_TEST
    TEST(ResultStatusDeathTest, TestStatusCtorStatusOk) {
        EXPECT_DEBUG_DEATH(
                {
                    // This will DCHECK
                    turbo::ResultStatus<int> thing(turbo::ok_status());
                    // In optimized mode, we are actually going to get error::INTERNAL for
                    // status here, rather than crashing, so check that.
                    EXPECT_FALSE(thing.ok());
                    EXPECT_EQ(thing.status().map_code(), turbo::kInternal);
                },
                "An OK status is not a valid constructor argument");
    }

    TEST(ResultStatusDeathTest, TestPointerStatusCtorStatusOk) {
        EXPECT_DEBUG_DEATH(
                {
                    turbo::ResultStatus<int *> thing(turbo::ok_status());
                    // In optimized mode, we are actually going to get error::INTERNAL for
                    // status here, rather than crashing, so check that.
                    EXPECT_FALSE(thing.ok());
                    EXPECT_EQ(thing.status().map_code(), turbo::kInternal);
                },
                "An OK status is not a valid constructor argument");
    }

#endif

    TEST(ResultStatus, ValueAccessor) {
        const int kIntValue = 110;
        {
            turbo::ResultStatus<int> status_or(kIntValue);
            EXPECT_EQ(kIntValue, status_or.value());
            EXPECT_EQ(kIntValue, std::move(status_or).value());
        }
        {
            turbo::ResultStatus<CopyDetector> status_or(kIntValue);
            EXPECT_THAT(status_or,
                        IsOkAndHolds(CopyDetectorHas(kIntValue, false, false)));
            CopyDetector copy_detector = status_or.value();
            EXPECT_THAT(copy_detector, CopyDetectorHas(kIntValue, false, true));
            copy_detector = std::move(status_or).value();
            EXPECT_THAT(copy_detector, CopyDetectorHas(kIntValue, true, false));
        }
    }

    TEST(ResultStatus, BadValueAccess) {
        const turbo::Status kError = turbo::cancelled_error("message");
        turbo::ResultStatus<int> status_or(kError);
        EXPECT_DEATH_OR_THROW(status_or.value(), kError);
    }

    TEST(ResultStatus, TestStatusCtor) {
        turbo::ResultStatus<int> thing(turbo::cancelled_error());
        EXPECT_FALSE(thing.ok());
        EXPECT_EQ(thing.status().map_code(), turbo::kCancelled);
    }

    TEST(ResultStatus, TestValueCtor) {
        const int kI = 4;
        const turbo::ResultStatus<int> thing(kI);
        EXPECT_TRUE(thing.ok());
        EXPECT_EQ(kI, *thing);
    }

    struct Foo {
        const int x;

        explicit Foo(int y) : x(y) {}
    };

    TEST(ResultStatus, InPlaceConstruction) {
        EXPECT_THAT(turbo::ResultStatus<Foo>(std::in_place, 10),
                    IsOkAndHolds(Field(&Foo::x, 10)));
    }

    struct InPlaceHelper {
        InPlaceHelper(std::initializer_list<int> xs, std::unique_ptr<int> yy)
                : x(xs), y(std::move(yy)) {}

        const std::vector<int> x;
        std::unique_ptr<int> y;
    };

    TEST(ResultStatus, InPlaceInitListConstruction) {
        turbo::ResultStatus<InPlaceHelper> status_or(std::in_place, {10, 11, 12},
                                                     std::make_unique<int>(13));
        EXPECT_THAT(status_or, IsOkAndHolds(AllOf(
                Field(&InPlaceHelper::x, ElementsAre(10, 11, 12)),
                Field(&InPlaceHelper::y, Pointee(13)))));
    }

    TEST(ResultStatus, Emplace) {
        turbo::ResultStatus<Foo> status_or_foo(10);
        status_or_foo.emplace(20);
        EXPECT_THAT(status_or_foo, IsOkAndHolds(Field(&Foo::x, 20)));
        status_or_foo = turbo::invalid_argument_error("msg");
        EXPECT_FALSE(status_or_foo.ok());
        EXPECT_EQ(status_or_foo.status().map_code(), turbo::kInvalidArgument);
        EXPECT_EQ(status_or_foo.status().message(), "msg");
        status_or_foo.emplace(20);
        EXPECT_THAT(status_or_foo, IsOkAndHolds(Field(&Foo::x, 20)));
    }

    TEST(ResultStatus, EmplaceInitializerList) {
        turbo::ResultStatus<InPlaceHelper> status_or(std::in_place, {10, 11, 12},
                                                     std::make_unique<int>(13));
        status_or.emplace({1, 2, 3}, std::make_unique<int>(4));
        EXPECT_THAT(status_or,
                    IsOkAndHolds(AllOf(Field(&InPlaceHelper::x, ElementsAre(1, 2, 3)),
                                       Field(&InPlaceHelper::y, Pointee(4)))));
        status_or = turbo::invalid_argument_error("msg");
        EXPECT_FALSE(status_or.ok());
        EXPECT_EQ(status_or.status().map_code(), turbo::kInvalidArgument);
        EXPECT_EQ(status_or.status().message(), "msg");
        status_or.emplace({1, 2, 3}, std::make_unique<int>(4));
        EXPECT_THAT(status_or,
                    IsOkAndHolds(AllOf(Field(&InPlaceHelper::x, ElementsAre(1, 2, 3)),
                                       Field(&InPlaceHelper::y, Pointee(4)))));
    }

    TEST(ResultStatus, TestCopyCtorStatusOk) {
        const int kI = 4;
        const turbo::ResultStatus<int> original(kI);
        const turbo::ResultStatus<int> copy(original);
        EXPECT_OK(copy.status());
        EXPECT_EQ(*original, *copy);
    }

    TEST(ResultStatus, TestCopyCtorStatusNotOk) {
        turbo::ResultStatus<int> original(turbo::cancelled_error());
        turbo::ResultStatus<int> copy(original);
        EXPECT_EQ(copy.status().map_code(), turbo::kCancelled);
    }

    TEST(ResultStatus, TestCopyCtorNonAssignable) {
        const int kI = 4;
        CopyNoAssign value(kI);
        turbo::ResultStatus<CopyNoAssign> original(value);
        turbo::ResultStatus<CopyNoAssign> copy(original);
        EXPECT_OK(copy.status());
        EXPECT_EQ(original->foo, copy->foo);
    }

    TEST(ResultStatus, TestCopyCtorStatusOKConverting) {
        const int kI = 4;
        turbo::ResultStatus<int> original(kI);
        turbo::ResultStatus<double> copy(original);
        EXPECT_OK(copy.status());
        EXPECT_DOUBLE_EQ(*original, *copy);
    }

    TEST(ResultStatus, TestCopyCtorStatusNotOkConverting) {
        turbo::ResultStatus<int> original(turbo::cancelled_error());
        turbo::ResultStatus<double> copy(original);
        EXPECT_EQ(copy.status(), original.status());
    }

    TEST(ResultStatus, TestAssignmentStatusOk) {
        // Copy assignmment
        {
            const auto p = std::make_shared<int>(17);
            turbo::ResultStatus<std::shared_ptr<int>> source(p);

            turbo::ResultStatus<std::shared_ptr<int>> target;
            target = source;

            ASSERT_TRUE(target.ok());
            EXPECT_OK(target.status());
            EXPECT_EQ(p, *target);

            ASSERT_TRUE(source.ok());
            EXPECT_OK(source.status());
            EXPECT_EQ(p, *source);
        }

        // Move asssignment
        {
            const auto p = std::make_shared<int>(17);
            turbo::ResultStatus<std::shared_ptr<int>> source(p);

            turbo::ResultStatus<std::shared_ptr<int>> target;
            target = std::move(source);

            ASSERT_TRUE(target.ok());
            EXPECT_OK(target.status());
            EXPECT_EQ(p, *target);

            ASSERT_TRUE(source.ok());
            EXPECT_OK(source.status());
            EXPECT_EQ(nullptr, *source);
        }
    }

    TEST(ResultStatus, TestAssignmentStatusNotOk) {
        // Copy assignment
        {
            const turbo::Status expected = turbo::cancelled_error();
            turbo::ResultStatus<int> source(expected);

            turbo::ResultStatus<int> target;
            target = source;

            EXPECT_FALSE(target.ok());
            EXPECT_EQ(expected, target.status());

            EXPECT_FALSE(source.ok());
            EXPECT_EQ(expected, source.status());
        }

        // Move assignment
        {
            const turbo::Status expected = turbo::cancelled_error();
            turbo::ResultStatus<int> source(expected);

            turbo::ResultStatus<int> target;
            target = std::move(source);

            EXPECT_FALSE(target.ok());
            EXPECT_EQ(expected, target.status());

            EXPECT_FALSE(source.ok());
            EXPECT_EQ(source.status().map_code(), turbo::kInternal);
        }
    }

    TEST(ResultStatus, TestAssignmentStatusOKConverting) {
        // Copy assignment
        {
            const int kI = 4;
            turbo::ResultStatus<int> source(kI);

            turbo::ResultStatus<double> target;
            target = source;

            ASSERT_TRUE(target.ok());
            EXPECT_OK(target.status());
            EXPECT_DOUBLE_EQ(kI, *target);

            ASSERT_TRUE(source.ok());
            EXPECT_OK(source.status());
            EXPECT_DOUBLE_EQ(kI, *source);
        }

        // Move assignment
        {
            const auto p = new int(17);
            turbo::ResultStatus<std::unique_ptr<int>> source(turbo::WrapUnique(p));

            turbo::ResultStatus<std::shared_ptr<int>> target;
            target = std::move(source);

            ASSERT_TRUE(target.ok());
            EXPECT_OK(target.status());
            EXPECT_EQ(p, target->get());

            ASSERT_TRUE(source.ok());
            EXPECT_OK(source.status());
            EXPECT_EQ(nullptr, source->get());
        }
    }

    struct A {
        int x;
    };

    struct ImplicitConstructibleFromA {
        int x;
        bool moved;

        ImplicitConstructibleFromA(const A &a)  // NOLINT
                : x(a.x), moved(false) {}

        ImplicitConstructibleFromA(A &&a)  // NOLINT
                : x(a.x), moved(true) {}
    };

    TEST(ResultStatus, ImplicitConvertingConstructor) {
        EXPECT_THAT(
                turbo::implicit_cast<turbo::ResultStatus<ImplicitConstructibleFromA>>(
                        turbo::ResultStatus<A>(A{11})),
                IsOkAndHolds(AllOf(Field(&ImplicitConstructibleFromA::x, 11),
                                   Field(&ImplicitConstructibleFromA::moved, true))));
        turbo::ResultStatus<A> a(A{12});
        EXPECT_THAT(
                turbo::implicit_cast<turbo::ResultStatus<ImplicitConstructibleFromA>>(a),
                IsOkAndHolds(AllOf(Field(&ImplicitConstructibleFromA::x, 12),
                                   Field(&ImplicitConstructibleFromA::moved, false))));
    }

    struct ExplicitConstructibleFromA {
        int x;
        bool moved;

        explicit ExplicitConstructibleFromA(const A &a) : x(a.x), moved(false) {}

        explicit ExplicitConstructibleFromA(A &&a) : x(a.x), moved(true) {}
    };

    TEST(ResultStatus, ExplicitConvertingConstructor) {
        EXPECT_FALSE(
                (std::is_convertible<const turbo::ResultStatus<A> &,
                        turbo::ResultStatus<ExplicitConstructibleFromA>>::value));
        EXPECT_FALSE(
                (std::is_convertible<turbo::ResultStatus<A> &&,
                        turbo::ResultStatus<ExplicitConstructibleFromA>>::value));
        EXPECT_THAT(
                turbo::ResultStatus<ExplicitConstructibleFromA>(turbo::ResultStatus<A>(A{11})),
                IsOkAndHolds(AllOf(Field(&ExplicitConstructibleFromA::x, 11),
                                   Field(&ExplicitConstructibleFromA::moved, true))));
        turbo::ResultStatus<A> a(A{12});
        EXPECT_THAT(
                turbo::ResultStatus<ExplicitConstructibleFromA>(a),
                IsOkAndHolds(AllOf(Field(&ExplicitConstructibleFromA::x, 12),
                                   Field(&ExplicitConstructibleFromA::moved, false))));
    }

    struct ImplicitConstructibleFromBool {
        ImplicitConstructibleFromBool(bool y) : x(y) {}  // NOLINT
        bool x = false;
    };

    struct ConvertibleToBool {
        explicit ConvertibleToBool(bool y) : x(y) {}

        operator bool() const { return x; }  // NOLINT
        bool x = false;
    };

    TEST(ResultStatus, ImplicitBooleanConstructionWithImplicitCasts) {
        EXPECT_THAT(turbo::ResultStatus<bool>(turbo::ResultStatus<ConvertibleToBool>(true)),
                    IsOkAndHolds(true));
        EXPECT_THAT(turbo::ResultStatus<bool>(turbo::ResultStatus<ConvertibleToBool>(false)),
                    IsOkAndHolds(false));
        EXPECT_THAT(
                turbo::implicit_cast<turbo::ResultStatus<ImplicitConstructibleFromBool>>(
                        turbo::ResultStatus<bool>(false)),
                IsOkAndHolds(Field(&ImplicitConstructibleFromBool::x, false)));
        EXPECT_FALSE((std::is_convertible<
                turbo::ResultStatus<ConvertibleToBool>,
                turbo::ResultStatus<ImplicitConstructibleFromBool>>::value));
    }

    TEST(ResultStatus, BooleanConstructionWithImplicitCasts) {
        EXPECT_THAT(turbo::ResultStatus<bool>(turbo::ResultStatus<ConvertibleToBool>(true)),
                    IsOkAndHolds(true));
        EXPECT_THAT(turbo::ResultStatus<bool>(turbo::ResultStatus<ConvertibleToBool>(false)),
                    IsOkAndHolds(false));
        EXPECT_THAT(
                turbo::ResultStatus<ImplicitConstructibleFromBool>{
                        turbo::ResultStatus<bool>(false)},
                IsOkAndHolds(Field(&ImplicitConstructibleFromBool::x, false)));
        EXPECT_THAT(
                turbo::ResultStatus<ImplicitConstructibleFromBool>{
                        turbo::ResultStatus<bool>(turbo::invalid_argument_error(""))},
                Not(IsOk()));

        EXPECT_THAT(
                turbo::ResultStatus<ImplicitConstructibleFromBool>{
                        turbo::ResultStatus<ConvertibleToBool>(ConvertibleToBool{false})},
                IsOkAndHolds(Field(&ImplicitConstructibleFromBool::x, false)));
        EXPECT_THAT(
                turbo::ResultStatus<ImplicitConstructibleFromBool>{
                        turbo::ResultStatus<ConvertibleToBool>(turbo::invalid_argument_error(""))},
                Not(IsOk()));
    }

    TEST(ResultStatus, ConstImplicitCast) {
        EXPECT_THAT(turbo::implicit_cast<turbo::ResultStatus<bool>>(
                turbo::ResultStatus<const bool>(true)),
                    IsOkAndHolds(true));
        EXPECT_THAT(turbo::implicit_cast<turbo::ResultStatus<bool>>(
                turbo::ResultStatus<const bool>(false)),
                    IsOkAndHolds(false));
        EXPECT_THAT(turbo::implicit_cast<turbo::ResultStatus<const bool>>(
                turbo::ResultStatus<bool>(true)),
                    IsOkAndHolds(true));
        EXPECT_THAT(turbo::implicit_cast<turbo::ResultStatus<const bool>>(
                turbo::ResultStatus<bool>(false)),
                    IsOkAndHolds(false));
        EXPECT_THAT(turbo::implicit_cast<turbo::ResultStatus<const std::string>>(
                turbo::ResultStatus<std::string>("foo")),
                    IsOkAndHolds("foo"));
        EXPECT_THAT(turbo::implicit_cast<turbo::ResultStatus<std::string>>(
                turbo::ResultStatus<const std::string>("foo")),
                    IsOkAndHolds("foo"));
        EXPECT_THAT(
                turbo::implicit_cast<turbo::ResultStatus<std::shared_ptr<const std::string>>>(
                        turbo::ResultStatus<std::shared_ptr<std::string>>(
                                std::make_shared<std::string>("foo"))),
                IsOkAndHolds(Pointee(std::string("foo"))));
    }

    TEST(ResultStatus, ConstExplicitConstruction) {
        EXPECT_THAT(turbo::ResultStatus<bool>(turbo::ResultStatus<const bool>(true)),
                    IsOkAndHolds(true));
        EXPECT_THAT(turbo::ResultStatus<bool>(turbo::ResultStatus<const bool>(false)),
                    IsOkAndHolds(false));
        EXPECT_THAT(turbo::ResultStatus<const bool>(turbo::ResultStatus<bool>(true)),
                    IsOkAndHolds(true));
        EXPECT_THAT(turbo::ResultStatus<const bool>(turbo::ResultStatus<bool>(false)),
                    IsOkAndHolds(false));
    }

    struct ExplicitConstructibleFromInt {
        int x;

        explicit ExplicitConstructibleFromInt(int y) : x(y) {}
    };

    TEST(ResultStatus, ExplicitConstruction) {
        EXPECT_THAT(turbo::ResultStatus<ExplicitConstructibleFromInt>(10),
                    IsOkAndHolds(Field(&ExplicitConstructibleFromInt::x, 10)));
    }

    TEST(ResultStatus, ImplicitConstruction) {
        // Check implicit casting works.
        auto status_or =
                turbo::implicit_cast<turbo::ResultStatus<std::variant<int, std::string>>>(10);
        EXPECT_THAT(status_or, IsOkAndHolds(VariantWith<int>(10)));
    }

    TEST(ResultStatus, ImplicitConstructionFromInitliazerList) {
        // Note: dropping the explicit std::initializer_list<int> is not supported
        // by turbo::ResultStatus or std::optional.
        auto status_or =
                turbo::implicit_cast<turbo::ResultStatus<std::vector<int>>>({{10, 20, 30}});
        EXPECT_THAT(status_or, IsOkAndHolds(ElementsAre(10, 20, 30)));
    }

    TEST(ResultStatus, UniquePtrImplicitConstruction) {
        auto status_or = turbo::implicit_cast<turbo::ResultStatus<std::unique_ptr<Base1>>>(
                std::make_unique<Derived>());
        EXPECT_THAT(status_or, IsOkAndHolds(Ne(nullptr)));
    }

    TEST(ResultStatus, NestedResultStatusCopyAndMoveConstructorTests) {
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> status_or = CopyDetector(10);
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> status_error =
                turbo::invalid_argument_error("foo");
        EXPECT_THAT(status_or,
                    IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, true, false))));
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> a = status_or;
        EXPECT_THAT(a, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, false, true))));
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> a_err = status_error;
        EXPECT_THAT(a_err, Not(IsOk()));

        const turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> &cref = status_or;
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> b = cref;  // NOLINT
        EXPECT_THAT(b, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, false, true))));
        const turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> &cref_err = status_error;
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> b_err = cref_err;  // NOLINT
        EXPECT_THAT(b_err, Not(IsOk()));

        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> c = std::move(status_or);
        EXPECT_THAT(c, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, true, false))));
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> c_err = std::move(status_error);
        EXPECT_THAT(c_err, Not(IsOk()));
    }

    TEST(ResultStatus, NestedResultStatusCopyAndMoveAssignment) {
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> status_or = CopyDetector(10);
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> status_error =
                turbo::invalid_argument_error("foo");
        turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> a;
        a = status_or;
        EXPECT_THAT(a, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, false, true))));
        a = status_error;
        EXPECT_THAT(a, Not(IsOk()));

        const turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> &cref = status_or;
        a = cref;
        EXPECT_THAT(a, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, false, true))));
        const turbo::ResultStatus<turbo::ResultStatus<CopyDetector>> &cref_err = status_error;
        a = cref_err;
        EXPECT_THAT(a, Not(IsOk()));
        a = std::move(status_or);
        EXPECT_THAT(a, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, true, false))));
        a = std::move(status_error);
        EXPECT_THAT(a, Not(IsOk()));
    }

    struct Copyable {
        Copyable() {}

        Copyable(const Copyable &) {}

        Copyable &operator=(const Copyable &) { return *this; }
    };

    struct MoveOnly {
        MoveOnly() {}

        MoveOnly(MoveOnly &&) {}

        MoveOnly &operator=(MoveOnly &&) { return *this; }
    };

    struct NonMovable {
        NonMovable() {}

        NonMovable(const NonMovable &) = delete;

        NonMovable(NonMovable &&) = delete;

        NonMovable &operator=(const NonMovable &) = delete;

        NonMovable &operator=(NonMovable &&) = delete;
    };

    TEST(ResultStatus, CopyAndMoveAbility) {
        EXPECT_TRUE(std::is_copy_constructible<Copyable>::value);
        EXPECT_TRUE(std::is_copy_assignable<Copyable>::value);
        EXPECT_TRUE(std::is_move_constructible<Copyable>::value);
        EXPECT_TRUE(std::is_move_assignable<Copyable>::value);
        EXPECT_FALSE(std::is_copy_constructible<MoveOnly>::value);
        EXPECT_FALSE(std::is_copy_assignable<MoveOnly>::value);
        EXPECT_TRUE(std::is_move_constructible<MoveOnly>::value);
        EXPECT_TRUE(std::is_move_assignable<MoveOnly>::value);
        EXPECT_FALSE(std::is_copy_constructible<NonMovable>::value);
        EXPECT_FALSE(std::is_copy_assignable<NonMovable>::value);
        EXPECT_FALSE(std::is_move_constructible<NonMovable>::value);
        EXPECT_FALSE(std::is_move_assignable<NonMovable>::value);
    }

    TEST(ResultStatus, ResultStatusAnyCopyAndMoveConstructorTests) {
        turbo::ResultStatus<std::any> status_or = CopyDetector(10);
        turbo::ResultStatus<std::any> status_error = turbo::invalid_argument_error("foo");
        EXPECT_THAT(
                status_or,
                IsOkAndHolds(AnyWith<CopyDetector>(CopyDetectorHas(10, true, false))));
        turbo::ResultStatus<std::any> a = status_or;
        EXPECT_THAT(
                a, IsOkAndHolds(AnyWith<CopyDetector>(CopyDetectorHas(10, false, true))));
        turbo::ResultStatus<std::any> a_err = status_error;
        EXPECT_THAT(a_err, Not(IsOk()));

        const turbo::ResultStatus<std::any> &cref = status_or;
        // No lint for no-change copy.
        turbo::ResultStatus<std::any> b = cref;  // NOLINT
        EXPECT_THAT(
                b, IsOkAndHolds(AnyWith<CopyDetector>(CopyDetectorHas(10, false, true))));
        const turbo::ResultStatus<std::any> &cref_err = status_error;
        // No lint for no-change copy.
        turbo::ResultStatus<std::any> b_err = cref_err;  // NOLINT
        EXPECT_THAT(b_err, Not(IsOk()));

        turbo::ResultStatus<std::any> c = std::move(status_or);
        EXPECT_THAT(
                c, IsOkAndHolds(AnyWith<CopyDetector>(CopyDetectorHas(10, true, false))));
        turbo::ResultStatus<std::any> c_err = std::move(status_error);
        EXPECT_THAT(c_err, Not(IsOk()));
    }

    TEST(ResultStatus, ResultStatusAnyCopyAndMoveAssignment) {
        turbo::ResultStatus<std::any> status_or = CopyDetector(10);
        turbo::ResultStatus<std::any> status_error = turbo::invalid_argument_error("foo");
        turbo::ResultStatus<std::any> a;
        a = status_or;
        EXPECT_THAT(
                a, IsOkAndHolds(AnyWith<CopyDetector>(CopyDetectorHas(10, false, true))));
        a = status_error;
        EXPECT_THAT(a, Not(IsOk()));

        const turbo::ResultStatus<std::any> &cref = status_or;
        a = cref;
        EXPECT_THAT(
                a, IsOkAndHolds(AnyWith<CopyDetector>(CopyDetectorHas(10, false, true))));
        const turbo::ResultStatus<std::any> &cref_err = status_error;
        a = cref_err;
        EXPECT_THAT(a, Not(IsOk()));
        a = std::move(status_or);
        EXPECT_THAT(
                a, IsOkAndHolds(AnyWith<CopyDetector>(CopyDetectorHas(10, true, false))));
        a = std::move(status_error);
        EXPECT_THAT(a, Not(IsOk()));
    }

    TEST(ResultStatus, ResultStatusCopyAndMoveTestsConstructor) {
        turbo::ResultStatus<CopyDetector> status_or(10);
        ASSERT_THAT(status_or, IsOkAndHolds(CopyDetectorHas(10, false, false)));
        turbo::ResultStatus<CopyDetector> a(status_or);
        EXPECT_THAT(a, IsOkAndHolds(CopyDetectorHas(10, false, true)));
        const turbo::ResultStatus<CopyDetector> &cref = status_or;
        turbo::ResultStatus<CopyDetector> b(cref);  // NOLINT
        EXPECT_THAT(b, IsOkAndHolds(CopyDetectorHas(10, false, true)));
        turbo::ResultStatus<CopyDetector> c(std::move(status_or));
        EXPECT_THAT(c, IsOkAndHolds(CopyDetectorHas(10, true, false)));
    }

    TEST(ResultStatus, ResultStatusCopyAndMoveTestsAssignment) {
        turbo::ResultStatus<CopyDetector> status_or(10);
        ASSERT_THAT(status_or, IsOkAndHolds(CopyDetectorHas(10, false, false)));
        turbo::ResultStatus<CopyDetector> a;
        a = status_or;
        EXPECT_THAT(a, IsOkAndHolds(CopyDetectorHas(10, false, true)));
        const turbo::ResultStatus<CopyDetector> &cref = status_or;
        turbo::ResultStatus<CopyDetector> b;
        b = cref;
        EXPECT_THAT(b, IsOkAndHolds(CopyDetectorHas(10, false, true)));
        turbo::ResultStatus<CopyDetector> c;
        c = std::move(status_or);
        EXPECT_THAT(c, IsOkAndHolds(CopyDetectorHas(10, true, false)));
    }

    TEST(ResultStatus, TurboAnyAssignment) {
        EXPECT_FALSE((std::is_assignable<turbo::ResultStatus<std::any>,
                turbo::ResultStatus<int>>::value));
        turbo::ResultStatus<std::any> status_or;
        status_or = turbo::invalid_argument_error("foo");
        EXPECT_THAT(status_or, Not(IsOk()));
    }

    TEST(ResultStatus, ImplicitAssignment) {
        turbo::ResultStatus<std::variant<int, std::string>> status_or;
        status_or = 10;
        EXPECT_THAT(status_or, IsOkAndHolds(VariantWith<int>(10)));
    }

    TEST(ResultStatus, SelfDirectInitAssignment) {
        turbo::ResultStatus<std::vector<int>> status_or = {{10, 20, 30}};
        status_or = *status_or;
        EXPECT_THAT(status_or, IsOkAndHolds(ElementsAre(10, 20, 30)));
    }

    TEST(ResultStatus, ImplicitCastFromInitializerList) {
        turbo::ResultStatus<std::vector<int>> status_or = {{10, 20, 30}};
        EXPECT_THAT(status_or, IsOkAndHolds(ElementsAre(10, 20, 30)));
    }

    TEST(ResultStatus, UniquePtrImplicitAssignment) {
        turbo::ResultStatus<std::unique_ptr<Base1>> status_or;
        status_or = std::make_unique<Derived>();
        EXPECT_THAT(status_or, IsOkAndHolds(Ne(nullptr)));
    }

    TEST(ResultStatus, Pointer) {
        struct A {
        };
        struct B : public A {
        };
        struct C : private A {
        };

        EXPECT_TRUE((std::is_constructible<turbo::ResultStatus<A *>, B *>::value));
        EXPECT_TRUE((std::is_convertible<B *, turbo::ResultStatus<A *>>::value));
        EXPECT_FALSE((std::is_constructible<turbo::ResultStatus<A *>, C *>::value));
        EXPECT_FALSE((std::is_convertible<C *, turbo::ResultStatus<A *>>::value));
    }

    TEST(ResultStatus, TestAssignmentStatusNotOkConverting) {
        // Copy assignment
        {
            const turbo::Status expected = turbo::cancelled_error();
            turbo::ResultStatus<int> source(expected);

            turbo::ResultStatus<double> target;
            target = source;

            EXPECT_FALSE(target.ok());
            EXPECT_EQ(expected, target.status());

            EXPECT_FALSE(source.ok());
            EXPECT_EQ(expected, source.status());
        }

        // Move assignment
        {
            const turbo::Status expected = turbo::cancelled_error();
            turbo::ResultStatus<int> source(expected);

            turbo::ResultStatus<double> target;
            target = std::move(source);

            EXPECT_FALSE(target.ok());
            EXPECT_EQ(expected, target.status());

            EXPECT_FALSE(source.ok());
            EXPECT_EQ(source.status().map_code(), turbo::kInternal);
        }
    }

    TEST(ResultStatus, SelfAssignment) {
        // Copy-assignment, status OK
        {
            // A string long enough that it's likely to defeat any inline representation
            // optimization.
            const std::string long_str(128, 'a');

            turbo::ResultStatus<std::string> so = long_str;
            so = *&so;

            ASSERT_TRUE(so.ok());
            EXPECT_OK(so.status());
            EXPECT_EQ(long_str, *so);
        }

        // Copy-assignment, error status
        {
            turbo::ResultStatus<int> so = turbo::not_found_error("taco");
            so = *&so;

            EXPECT_FALSE(so.ok());
            EXPECT_EQ(so.status().map_code(), turbo::kNotFound);
            EXPECT_EQ(so.status().message(), "taco");
        }

        // Move-assignment with copyable type, status OK
        {
            turbo::ResultStatus<int> so = 17;

            // Fool the compiler, which otherwise complains.
            auto &same = so;
            so = std::move(same);

            ASSERT_TRUE(so.ok());
            EXPECT_OK(so.status());
            EXPECT_EQ(17, *so);
        }

        // Move-assignment with copyable type, error status
        {
            turbo::ResultStatus<int> so = turbo::not_found_error("taco");

            // Fool the compiler, which otherwise complains.
            auto &same = so;
            so = std::move(same);

            EXPECT_FALSE(so.ok());
            EXPECT_EQ(so.status().map_code(), turbo::kNotFound);
            EXPECT_EQ(so.status().message(), "taco");
        }

        // Move-assignment with non-copyable type, status OK
        {
            const auto raw = new int(17);
            turbo::ResultStatus<std::unique_ptr<int>> so = turbo::WrapUnique(raw);

            // Fool the compiler, which otherwise complains.
            auto &same = so;
            so = std::move(same);

            ASSERT_TRUE(so.ok());
            EXPECT_OK(so.status());
            EXPECT_EQ(raw, so->get());
        }

        // Move-assignment with non-copyable type, error status
        {
            turbo::ResultStatus<std::unique_ptr<int>> so = turbo::not_found_error("taco");

            // Fool the compiler, which otherwise complains.
            auto &same = so;
            so = std::move(same);

            EXPECT_FALSE(so.ok());
            EXPECT_EQ(so.status().map_code(), turbo::kNotFound);
            EXPECT_EQ(so.status().message(), "taco");
        }
    }

// These types form the overload sets of the constructors and the assignment
// operators of `MockValue`. They distinguish construction from assignment,
// lvalue from rvalue.
    struct FromConstructibleAssignableLvalue {
    };
    struct FromConstructibleAssignableRvalue {
    };
    struct FromImplicitConstructibleOnly {
    };
    struct FromAssignableOnly {
    };

// This class is for testing the forwarding value assignments of `ResultStatus`.
// `from_rvalue` indicates whether the constructor or the assignment taking
// rvalue reference is called. `from_assignment` indicates whether any
// assignment is called.
    struct MockValue {
        // Constructs `MockValue` from `FromConstructibleAssignableLvalue`.
        MockValue(const FromConstructibleAssignableLvalue &)  // NOLINT
                : from_rvalue(false), assigned(false) {}

        // Constructs `MockValue` from `FromConstructibleAssignableRvalue`.
        MockValue(FromConstructibleAssignableRvalue &&)  // NOLINT
                : from_rvalue(true), assigned(false) {}

        // Constructs `MockValue` from `FromImplicitConstructibleOnly`.
        // `MockValue` is not assignable from `FromImplicitConstructibleOnly`.
        MockValue(const FromImplicitConstructibleOnly &)  // NOLINT
                : from_rvalue(false), assigned(false) {}

        // Assigns `FromConstructibleAssignableLvalue`.
        MockValue &operator=(const FromConstructibleAssignableLvalue &) {
            from_rvalue = false;
            assigned = true;
            return *this;
        }

        // Assigns `FromConstructibleAssignableRvalue` (rvalue only).
        MockValue &operator=(FromConstructibleAssignableRvalue &&) {
            from_rvalue = true;
            assigned = true;
            return *this;
        }

        // Assigns `FromAssignableOnly`, but not constructible from
        // `FromAssignableOnly`.
        MockValue &operator=(const FromAssignableOnly &) {
            from_rvalue = false;
            assigned = true;
            return *this;
        }

        bool from_rvalue;
        bool assigned;
    };

// operator=(U&&)
    TEST(ResultStatus, PerfectForwardingAssignment) {
        // U == T
        constexpr int kValue1 = 10, kValue2 = 20;
        turbo::ResultStatus<CopyDetector> status_or;
        CopyDetector lvalue(kValue1);
        status_or = lvalue;
        EXPECT_THAT(status_or, IsOkAndHolds(CopyDetectorHas(kValue1, false, true)));
        status_or = CopyDetector(kValue2);
        EXPECT_THAT(status_or, IsOkAndHolds(CopyDetectorHas(kValue2, true, false)));

        // U != T
        EXPECT_TRUE(
                (std::is_assignable<turbo::ResultStatus<MockValue> &,
                        const FromConstructibleAssignableLvalue &>::value));
        EXPECT_TRUE((std::is_assignable<turbo::ResultStatus<MockValue> &,
                FromConstructibleAssignableLvalue &&>::value));
        EXPECT_FALSE(
                (std::is_assignable<turbo::ResultStatus<MockValue> &,
                        const FromConstructibleAssignableRvalue &>::value));
        EXPECT_TRUE((std::is_assignable<turbo::ResultStatus<MockValue> &,
                FromConstructibleAssignableRvalue &&>::value));
        EXPECT_TRUE(
                (std::is_assignable<turbo::ResultStatus<MockValue> &,
                        const FromImplicitConstructibleOnly &>::value));
        EXPECT_FALSE((std::is_assignable<turbo::ResultStatus<MockValue> &,
                const FromAssignableOnly &>::value));

        turbo::ResultStatus<MockValue> from_lvalue(FromConstructibleAssignableLvalue{});
        EXPECT_FALSE(from_lvalue->from_rvalue);
        EXPECT_FALSE(from_lvalue->assigned);
        from_lvalue = FromConstructibleAssignableLvalue{};
        EXPECT_FALSE(from_lvalue->from_rvalue);
        EXPECT_TRUE(from_lvalue->assigned);

        turbo::ResultStatus<MockValue> from_rvalue(FromConstructibleAssignableRvalue{});
        EXPECT_TRUE(from_rvalue->from_rvalue);
        EXPECT_FALSE(from_rvalue->assigned);
        from_rvalue = FromConstructibleAssignableRvalue{};
        EXPECT_TRUE(from_rvalue->from_rvalue);
        EXPECT_TRUE(from_rvalue->assigned);

        turbo::ResultStatus<MockValue> from_implicit_constructible(
                FromImplicitConstructibleOnly{});
        EXPECT_FALSE(from_implicit_constructible->from_rvalue);
        EXPECT_FALSE(from_implicit_constructible->assigned);
        // construct a temporary `ResultStatus` object and invoke the `ResultStatus` move
        // assignment operator.
        from_implicit_constructible = FromImplicitConstructibleOnly{};
        EXPECT_FALSE(from_implicit_constructible->from_rvalue);
        EXPECT_FALSE(from_implicit_constructible->assigned);
    }

    TEST(ResultStatus, TestStatus) {
        turbo::ResultStatus<int> good(4);
        EXPECT_TRUE(good.ok());
        turbo::ResultStatus<int> bad(turbo::cancelled_error());
        EXPECT_FALSE(bad.ok());
        EXPECT_EQ(bad.status().map_code(), turbo::kCancelled);
    }

    TEST(ResultStatus, OperatorStarRefQualifiers) {
        static_assert(
                std::is_same<const int &,
                        decltype(*std::declval<const turbo::ResultStatus<int> &>())>(),
                "Unexpected ref-qualifiers");
        static_assert(
                std::is_same<int &, decltype(*std::declval<turbo::ResultStatus<int> &>())>(),
                "Unexpected ref-qualifiers");
        static_assert(
                std::is_same<const int &&,
                        decltype(*std::declval<const turbo::ResultStatus<int> &&>())>(),
                "Unexpected ref-qualifiers");
        static_assert(
                std::is_same<int &&, decltype(*std::declval<turbo::ResultStatus<int> &&>())>(),
                "Unexpected ref-qualifiers");
    }

    TEST(ResultStatus, OperatorStar) {
        const turbo::ResultStatus<std::string> const_lvalue("hello");
        EXPECT_EQ("hello", *const_lvalue);

        turbo::ResultStatus<std::string> lvalue("hello");
        EXPECT_EQ("hello", *lvalue);

        // Note: Recall that std::move() is equivalent to a static_cast to an rvalue
        // reference type.
        const turbo::ResultStatus<std::string> const_rvalue("hello");
        EXPECT_EQ("hello", *std::move(const_rvalue));  // NOLINT

        turbo::ResultStatus<std::string> rvalue("hello");
        EXPECT_EQ("hello", *std::move(rvalue));
    }

    TEST(ResultStatus, OperatorArrowQualifiers) {
        static_assert(
                std::is_same<
                        const int *,
                        decltype(std::declval<const turbo::ResultStatus<int> &>().operator->())>(),
                "Unexpected qualifiers");
        static_assert(
                std::is_same<
                        int *, decltype(std::declval<turbo::ResultStatus<int> &>().operator->())>(),
                "Unexpected qualifiers");
        static_assert(
                std::is_same<
                        const int *,
                        decltype(std::declval<const turbo::ResultStatus<int> &&>().operator->())>(),
                "Unexpected qualifiers");
        static_assert(
                std::is_same<
                        int *, decltype(std::declval<turbo::ResultStatus<int> &&>().operator->())>(),
                "Unexpected qualifiers");
    }

    TEST(ResultStatus, OperatorArrow) {
        const turbo::ResultStatus<std::string> const_lvalue("hello");
        EXPECT_EQ(std::string("hello"), const_lvalue->c_str());

        turbo::ResultStatus<std::string> lvalue("hello");
        EXPECT_EQ(std::string("hello"), lvalue->c_str());
    }

    TEST(ResultStatus, RValueStatus) {
        turbo::ResultStatus<int> so(turbo::not_found_error("taco"));
        const turbo::Status s = std::move(so).status();

        EXPECT_EQ(s.map_code(), turbo::kNotFound);
        EXPECT_EQ(s.message(), "taco");

        // Check that !ok() still implies !status().ok(), even after moving out of the
        // object. See the note on the rvalue ref-qualified status method.
        EXPECT_FALSE(so.ok());  // NOLINT
        EXPECT_FALSE(so.status().ok());
        EXPECT_EQ(so.status().map_code(), turbo::kInternal);
        EXPECT_EQ(so.status().message(), "Status accessed after move.");
    }

    TEST(ResultStatus, TestValue) {
        const int kI = 4;
        turbo::ResultStatus<int> thing(kI);
        EXPECT_EQ(kI, *thing);
    }

    TEST(ResultStatus, TestValueConst) {
        const int kI = 4;
        const turbo::ResultStatus<int> thing(kI);
        EXPECT_EQ(kI, *thing);
    }

    TEST(ResultStatus, TestPointerDefaultCtor) {
        turbo::ResultStatus<int *> thing;
        EXPECT_FALSE(thing.ok());
        EXPECT_EQ(thing.status().map_code(), turbo::kUnknown);
    }

    TEST(ResultStatus, TestPointerStatusCtor) {
        turbo::ResultStatus<int *> thing(turbo::cancelled_error());
        EXPECT_FALSE(thing.ok());
        EXPECT_EQ(thing.status().map_code(), turbo::kCancelled);
    }

    TEST(ResultStatus, TestPointerValueCtor) {
        const int kI = 4;

        // Construction from a non-null pointer
        {
            turbo::ResultStatus<const int *> so(&kI);
            EXPECT_TRUE(so.ok());
            EXPECT_OK(so.status());
            EXPECT_EQ(&kI, *so);
        }

        // Construction from a null pointer constant
        {
            turbo::ResultStatus<const int *> so(nullptr);
            EXPECT_TRUE(so.ok());
            EXPECT_OK(so.status());
            EXPECT_EQ(nullptr, *so);
        }

        // Construction from a non-literal null pointer
        {
            const int *const p = nullptr;

            turbo::ResultStatus<const int *> so(p);
            EXPECT_TRUE(so.ok());
            EXPECT_OK(so.status());
            EXPECT_EQ(nullptr, *so);
        }
    }

    TEST(ResultStatus, TestPointerCopyCtorStatusOk) {
        const int kI = 0;
        turbo::ResultStatus<const int *> original(&kI);
        turbo::ResultStatus<const int *> copy(original);
        EXPECT_OK(copy.status());
        EXPECT_EQ(*original, *copy);
    }

    TEST(ResultStatus, TestPointerCopyCtorStatusNotOk) {
        turbo::ResultStatus<int *> original(turbo::cancelled_error());
        turbo::ResultStatus<int *> copy(original);
        EXPECT_EQ(copy.status().map_code(), turbo::kCancelled);
    }

    TEST(ResultStatus, TestPointerCopyCtorStatusOKConverting) {
        Derived derived;
        turbo::ResultStatus<Derived *> original(&derived);
        turbo::ResultStatus<Base2 *> copy(original);
        EXPECT_OK(copy.status());
        EXPECT_EQ(static_cast<const Base2 *>(*original), *copy);
    }

    TEST(ResultStatus, TestPointerCopyCtorStatusNotOkConverting) {
        turbo::ResultStatus<Derived *> original(turbo::cancelled_error());
        turbo::ResultStatus<Base2 *> copy(original);
        EXPECT_EQ(copy.status().map_code(), turbo::kCancelled);
    }

    TEST(ResultStatus, TestPointerAssignmentStatusOk) {
        const int kI = 0;
        turbo::ResultStatus<const int *> source(&kI);
        turbo::ResultStatus<const int *> target;
        target = source;
        EXPECT_OK(target.status());
        EXPECT_EQ(*source, *target);
    }

    TEST(ResultStatus, TestPointerAssignmentStatusNotOk) {
        turbo::ResultStatus<int *> source(turbo::cancelled_error());
        turbo::ResultStatus<int *> target;
        target = source;
        EXPECT_EQ(target.status().map_code(), turbo::kCancelled);
    }

    TEST(ResultStatus, TestPointerAssignmentStatusOKConverting) {
        Derived derived;
        turbo::ResultStatus<Derived *> source(&derived);
        turbo::ResultStatus<Base2 *> target;
        target = source;
        EXPECT_OK(target.status());
        EXPECT_EQ(static_cast<const Base2 *>(*source), *target);
    }

    TEST(ResultStatus, TestPointerAssignmentStatusNotOkConverting) {
        turbo::ResultStatus<Derived *> source(turbo::cancelled_error());
        turbo::ResultStatus<Base2 *> target;
        target = source;
        EXPECT_EQ(target.status(), source.status());
    }

    TEST(ResultStatus, TestPointerStatus) {
        const int kI = 0;
        turbo::ResultStatus<const int *> good(&kI);
        EXPECT_TRUE(good.ok());
        turbo::ResultStatus<const int *> bad(turbo::cancelled_error());
        EXPECT_EQ(bad.status().map_code(), turbo::kCancelled);
    }

    TEST(ResultStatus, TestPointerValue) {
        const int kI = 0;
        turbo::ResultStatus<const int *> thing(&kI);
        EXPECT_EQ(&kI, *thing);
    }

    TEST(ResultStatus, TestPointerValueConst) {
        const int kI = 0;
        const turbo::ResultStatus<const int *> thing(&kI);
        EXPECT_EQ(&kI, *thing);
    }

    TEST(ResultStatus, ResultStatusVectorOfUniquePointerCanReserveAndResize) {
        using EvilType = std::vector<std::unique_ptr<int>>;
        static_assert(std::is_copy_constructible<EvilType>::value, "");
        std::vector<::turbo::ResultStatus<EvilType>> v(5);
        v.reserve(v.capacity() + 10);
        v.resize(v.capacity() + 10);
    }

    TEST(ResultStatus, ConstPayload) {
        // A reduced version of a problematic type found in the wild. All of the
        // operations below should compile.
        turbo::ResultStatus<const int> a;

        // Copy-construction
        turbo::ResultStatus<const int> b(a);

        // Copy-assignment
        EXPECT_FALSE(std::is_copy_assignable<turbo::ResultStatus<const int>>::value);

        // Move-construction
        turbo::ResultStatus<const int> c(std::move(a));

        // Move-assignment
        EXPECT_FALSE(std::is_move_assignable<turbo::ResultStatus<const int>>::value);
    }

    TEST(ResultStatus, MapToResultStatusUniquePtr) {
        // A reduced version of a problematic type found in the wild. All of the
        // operations below should compile.
        using MapType = std::map<std::string, turbo::ResultStatus<std::unique_ptr<int>>>;

        MapType a;

        // Move-construction
        MapType b(std::move(a));

        // Move-assignment
        a = std::move(b);
    }

    TEST(ResultStatus, ValueOrOk) {
        const turbo::ResultStatus<int> status_or = 0;
        EXPECT_EQ(status_or.value_or(-1), 0);
    }

    TEST(ResultStatus, ValueOrDefault) {
        const turbo::ResultStatus<int> status_or = turbo::cancelled_error();
        EXPECT_EQ(status_or.value_or(-1), -1);
    }

    TEST(ResultStatus, MoveOnlyValueOrOk) {
        EXPECT_THAT(turbo::ResultStatus<std::unique_ptr<int>>(std::make_unique<int>(0))
                            .value_or(std::make_unique<int>(-1)),
                    Pointee(0));
    }

    TEST(ResultStatus, MoveOnlyValueOrDefault) {
        EXPECT_THAT(turbo::ResultStatus<std::unique_ptr<int>>(turbo::cancelled_error())
                            .value_or(std::make_unique<int>(-1)),
                    Pointee(-1));
    }

    static turbo::ResultStatus<int> make_status() { return 100; }

    TEST(ResultStatus, TestIgnoreError) { make_status().IgnoreError(); }

    TEST(ResultStatus, EqualityOperator) {
        constexpr size_t kNumCases = 4;
        std::array<turbo::ResultStatus<int>, kNumCases> group1 = {
                turbo::ResultStatus<int>(1), turbo::ResultStatus<int>(2),
                turbo::ResultStatus<int>(turbo::invalid_argument_error("msg")),
                turbo::ResultStatus<int>(turbo::internal_error("msg"))};
        std::array<turbo::ResultStatus<int>, kNumCases> group2 = {
                turbo::ResultStatus<int>(1), turbo::ResultStatus<int>(2),
                turbo::ResultStatus<int>(turbo::invalid_argument_error("msg")),
                turbo::ResultStatus<int>(turbo::internal_error("msg"))};
        for (size_t i = 0; i < kNumCases; ++i) {
            for (size_t j = 0; j < kNumCases; ++j) {
                if (i == j) {
                    EXPECT_TRUE(group1[i] == group2[j]);
                    EXPECT_FALSE(group1[i] != group2[j]);
                } else {
                    EXPECT_FALSE(group1[i] == group2[j]);
                    EXPECT_TRUE(group1[i] != group2[j]);
                }
            }
        }
    }

    struct MyType {
        bool operator==(const MyType &) const { return true; }
    };

    enum class ConvTraits {
        kNone = 0, kImplicit = 1, kExplicit = 2
    };

// This class has conversion operator to `ResultStatus<T>` based on value of
// `conv_traits`.
    template<typename T, ConvTraits conv_traits = ConvTraits::kNone>
    struct ResultStatusConversionBase {
    };

    template<typename T>
    struct ResultStatusConversionBase<T, ConvTraits::kImplicit> {
        operator turbo::ResultStatus<T>() const & {  // NOLINT
            return turbo::invalid_argument_error("conversion to turbo::ResultStatus");
        }

        operator turbo::ResultStatus<T>() && {  // NOLINT
            return turbo::invalid_argument_error("conversion to turbo::ResultStatus");
        }
    };

    template<typename T>
    struct ResultStatusConversionBase<T, ConvTraits::kExplicit> {
        explicit operator turbo::ResultStatus<T>() const & {
            return turbo::invalid_argument_error("conversion to turbo::ResultStatus");
        }

        explicit operator turbo::ResultStatus<T>() && {
            return turbo::invalid_argument_error("conversion to turbo::ResultStatus");
        }
    };

// This class has conversion operator to `T` based on the value of
// `conv_traits`.
    template<typename T, ConvTraits conv_traits = ConvTraits::kNone>
    struct ConversionBase {
    };

    template<typename T>
    struct ConversionBase<T, ConvTraits::kImplicit> {
        operator T() const & { return t; }         // NOLINT
        operator T() && { return std::move(t); }  // NOLINT
        T t;
    };

    template<typename T>
    struct ConversionBase<T, ConvTraits::kExplicit> {
        explicit operator T() const & { return t; }

        explicit operator T() && { return std::move(t); }

        T t;
    };

// This class has conversion operator to `turbo::Status` based on the value of
// `conv_traits`.
    template<ConvTraits conv_traits = ConvTraits::kNone>
    struct StatusConversionBase {
    };

    template<>
    struct StatusConversionBase<ConvTraits::kImplicit> {
        operator turbo::Status() const & {  // NOLINT
            return turbo::internal_error("conversion to Status");
        }

        operator turbo::Status() && {  // NOLINT
            return turbo::internal_error("conversion to Status");
        }
    };

    template<>
    struct StatusConversionBase<ConvTraits::kExplicit> {
        explicit operator turbo::Status() const & {  // NOLINT
            return turbo::internal_error("conversion to Status");
        }

        explicit operator turbo::Status() && {  // NOLINT
            return turbo::internal_error("conversion to Status");
        }
    };

    static constexpr int kConvToStatus = 1;
    static constexpr int kConvToResultStatus = 2;
    static constexpr int kConvToT = 4;
    static constexpr int kConvExplicit = 8;

    constexpr ConvTraits GetConvTraits(int bit, int config) {
        return (config & bit) == 0
               ? ConvTraits::kNone
               : ((config & kConvExplicit) == 0 ? ConvTraits::kImplicit
                                                : ConvTraits::kExplicit);
    }

// This class conditionally has conversion operator to `turbo::Status`, `T`,
// `ResultStatus<T>`, based on values of the template parameters.
    template<typename T, int config>
    struct CustomType
            : ResultStatusConversionBase<T, GetConvTraits(kConvToResultStatus, config)>,
              ConversionBase<T, GetConvTraits(kConvToT, config)>,
              StatusConversionBase<GetConvTraits(kConvToStatus, config)> {
    };

    struct ConvertibleToAnyResultStatus {
        template<typename T>
        operator turbo::ResultStatus<T>() const {  // NOLINT
            return turbo::invalid_argument_error("Conversion to turbo::ResultStatus");
        }
    };

// Test the rank of overload resolution for `ResultStatus<T>` constructor and
// assignment, from highest to lowest:
// 1. T/Status
// 2. U that has conversion operator to turbo::ResultStatus<T>
// 3. U that is convertible to Status
// 4. U that is convertible to T
    TEST(ResultStatus, ConstructionFromT) {
        // Construct turbo::ResultStatus<T> from T when T is convertible to
        // turbo::ResultStatus<T>
        {
            ConvertibleToAnyResultStatus v;
            turbo::ResultStatus<ConvertibleToAnyResultStatus> rtstatus(v);
            EXPECT_TRUE(rtstatus.ok());
        }
        {
            ConvertibleToAnyResultStatus v;
            turbo::ResultStatus<ConvertibleToAnyResultStatus> rtstatus = v;
            EXPECT_TRUE(rtstatus.ok());
        }
        // Construct turbo::ResultStatus<T> from T when T is explicitly convertible to
        // Status
        {
            CustomType<MyType, kConvToStatus | kConvExplicit> v;
            turbo::ResultStatus<CustomType<MyType, kConvToStatus | kConvExplicit>> rtstatus(
                    v);
            EXPECT_TRUE(rtstatus.ok());
        }
        {
            CustomType<MyType, kConvToStatus | kConvExplicit> v;
            turbo::ResultStatus<CustomType<MyType, kConvToStatus | kConvExplicit>> rtstatus =
                    v;
            EXPECT_TRUE(rtstatus.ok());
        }
    }

// Construct turbo::ResultStatus<T> from U when U is explicitly convertible to T
    TEST(ResultStatus, ConstructionFromTypeConvertibleToT) {
        {
            CustomType<MyType, kConvToT | kConvExplicit> v;
            turbo::ResultStatus<MyType> rtstatus(v);
            EXPECT_TRUE(rtstatus.ok());
        }
        {
            CustomType<MyType, kConvToT> v;
            turbo::ResultStatus<MyType> rtstatus = v;
            EXPECT_TRUE(rtstatus.ok());
        }
    }

// Construct turbo::ResultStatus<T> from U when U has explicit conversion operator to
// turbo::ResultStatus<T>
    TEST(ResultStatus, ConstructionFromTypeWithConversionOperatorToResultStatusT) {
        {
            CustomType<MyType, kConvToResultStatus | kConvExplicit> v;
            turbo::ResultStatus<MyType> rtstatus(v);
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToT | kConvToResultStatus | kConvExplicit> v;
            turbo::ResultStatus<MyType> rtstatus(v);
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToResultStatus | kConvToStatus | kConvExplicit> v;
            turbo::ResultStatus<MyType> rtstatus(v);
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType,
                    kConvToT | kConvToResultStatus | kConvToStatus | kConvExplicit>
                    v;
            turbo::ResultStatus<MyType> rtstatus(v);
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToResultStatus> v;
            turbo::ResultStatus<MyType> rtstatus = v;
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToT | kConvToResultStatus> v;
            turbo::ResultStatus<MyType> rtstatus = v;
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToResultStatus | kConvToStatus> v;
            turbo::ResultStatus<MyType> rtstatus = v;
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToT | kConvToResultStatus | kConvToStatus> v;
            turbo::ResultStatus<MyType> rtstatus = v;
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
    }

    TEST(ResultStatus, ConstructionFromTypeConvertibleToStatus) {
        // Construction fails because conversion to `Status` is explicit.
        {
            CustomType<MyType, kConvToStatus | kConvExplicit> v;
            turbo::ResultStatus<MyType> rtstatus(v);
            EXPECT_FALSE(rtstatus.ok());
            EXPECT_EQ(rtstatus.status(), static_cast<turbo::Status>(v));
        }
        {
            CustomType<MyType, kConvToT | kConvToStatus | kConvExplicit> v;
            turbo::ResultStatus<MyType> rtstatus(v);
            EXPECT_FALSE(rtstatus.ok());
            EXPECT_EQ(rtstatus.status(), static_cast<turbo::Status>(v));
        }
        {
            CustomType<MyType, kConvToStatus> v;
            turbo::ResultStatus<MyType> rtstatus = v;
            EXPECT_FALSE(rtstatus.ok());
            EXPECT_EQ(rtstatus.status(), static_cast<turbo::Status>(v));
        }
        {
            CustomType<MyType, kConvToT | kConvToStatus> v;
            turbo::ResultStatus<MyType> rtstatus = v;
            EXPECT_FALSE(rtstatus.ok());
            EXPECT_EQ(rtstatus.status(), static_cast<turbo::Status>(v));
        }
    }

    TEST(ResultStatus, AssignmentFromT) {
        // Assign to turbo::ResultStatus<T> from T when T is convertible to
        // turbo::ResultStatus<T>
        {
            ConvertibleToAnyResultStatus v;
            turbo::ResultStatus<ConvertibleToAnyResultStatus> rtstatus;
            rtstatus = v;
            EXPECT_TRUE(rtstatus.ok());
        }
        // Assign to turbo::ResultStatus<T> from T when T is convertible to Status
        {
            CustomType<MyType, kConvToStatus> v;
            turbo::ResultStatus<CustomType<MyType, kConvToStatus>> rtstatus;
            rtstatus = v;
            EXPECT_TRUE(rtstatus.ok());
        }
    }

    TEST(ResultStatus, AssignmentFromTypeConvertibleToT) {
        // Assign to turbo::ResultStatus<T> from U when U is convertible to T
        {
            CustomType<MyType, kConvToT> v;
            turbo::ResultStatus<MyType> rtstatus;
            rtstatus = v;
            EXPECT_TRUE(rtstatus.ok());
        }
    }

    TEST(ResultStatus, AssignmentFromTypeWithConversionOperatortoResultStatusT) {
        // Assign to turbo::ResultStatus<T> from U when U has conversion operator to
        // turbo::ResultStatus<T>
        {
            CustomType<MyType, kConvToResultStatus> v;
            turbo::ResultStatus<MyType> rtstatus;
            rtstatus = v;
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToT | kConvToResultStatus> v;
            turbo::ResultStatus<MyType> rtstatus;
            rtstatus = v;
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToResultStatus | kConvToStatus> v;
            turbo::ResultStatus<MyType> rtstatus;
            rtstatus = v;
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
        {
            CustomType<MyType, kConvToT | kConvToResultStatus | kConvToStatus> v;
            turbo::ResultStatus<MyType> rtstatus;
            rtstatus = v;
            EXPECT_EQ(rtstatus, v.operator turbo::ResultStatus<MyType>());
        }
    }

    TEST(ResultStatus, AssignmentFromTypeConvertibleToStatus) {
        // Assign to turbo::ResultStatus<T> from U when U is convertible to Status
        {
            CustomType<MyType, kConvToStatus> v;
            turbo::ResultStatus<MyType> rtstatus;
            rtstatus = v;
            EXPECT_FALSE(rtstatus.ok());
            EXPECT_EQ(rtstatus.status(), static_cast<turbo::Status>(v));
        }
        {
            CustomType<MyType, kConvToT | kConvToStatus> v;
            turbo::ResultStatus<MyType> rtstatus;
            rtstatus = v;
            EXPECT_FALSE(rtstatus.ok());
            EXPECT_EQ(rtstatus.status(), static_cast<turbo::Status>(v));
        }
    }

}  // namespace
