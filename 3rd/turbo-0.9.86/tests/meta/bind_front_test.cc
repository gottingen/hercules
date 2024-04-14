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

#include "turbo/meta/bind_front.h"

#include <stddef.h>

#include <functional>
#include <memory>
#include <string>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/memory/memory.h"

namespace {

    char CharAt(const char *s, size_t index) { return s[index]; }

    TEST_CASE("BindTest, Basics") {
        CHECK_EQ('C', turbo::bind_front(CharAt)("ABC", 2));
        CHECK_EQ('C', turbo::bind_front(CharAt, "ABC")(2));
        CHECK_EQ('C', turbo::bind_front(CharAt, "ABC", 2)());
    }

    TEST_CASE("BindTest, Lambda") {
        auto lambda = [](int x, int y, int z) { return x + y + z; };
        CHECK_EQ(6, turbo::bind_front(lambda)(1, 2, 3));
        CHECK_EQ(6, turbo::bind_front(lambda, 1)(2, 3));
        CHECK_EQ(6, turbo::bind_front(lambda, 1, 2)(3));
        CHECK_EQ(6, turbo::bind_front(lambda, 1, 2, 3)());
    }

    struct Functor {
        std::string operator()() &{ return "&"; }

        std::string operator()() const &{ return "const&"; }

        std::string operator()() &&{ return "&&"; }

        std::string operator()() const &&{ return "const&&"; }
    };

    TEST_CASE("BindTest, PerfectForwardingOfBoundArgs") {
        auto f = turbo::bind_front(Functor());
        const auto &cf = f;
        CHECK_EQ("&", f());
        CHECK_EQ("const&", cf());
        CHECK_EQ("&&", std::move(f)());
        CHECK_EQ("const&&", std::move(cf)());
    }

    struct ArgDescribe {
        std::string operator()(int &) const { return "&"; }             // NOLINT
        std::string operator()(const int &) const { return "const&"; }  // NOLINT
        std::string operator()(int &&) const { return "&&"; }

        std::string operator()(const int &&) const { return "const&&"; }
    };

    TEST_CASE("BindTest, PerfectForwardingOfFreeArgs") {
        ArgDescribe f;
        int i;
        CHECK_EQ("&", turbo::bind_front(f)(static_cast<int &>(i)));
        CHECK_EQ("const&", turbo::bind_front(f)(static_cast<const int &>(i)));
        CHECK_EQ("&&", turbo::bind_front(f)(static_cast<int &&>(i)));
        CHECK_EQ("const&&", turbo::bind_front(f)(static_cast<const int &&>(i)));
    }

    struct NonCopyableFunctor {
        NonCopyableFunctor() = default;

        NonCopyableFunctor(const NonCopyableFunctor &) = delete;

        NonCopyableFunctor &operator=(const NonCopyableFunctor &) = delete;

        const NonCopyableFunctor *operator()() const { return this; }
    };

    TEST_CASE("BindTest, RefToFunctor") {
        // It won't copy/move the functor and use the original object.
        NonCopyableFunctor ncf;
        auto bound_ncf = turbo::bind_front(std::ref(ncf));
        auto bound_ncf_copy = bound_ncf;
        CHECK_EQ(&ncf, bound_ncf_copy());
    }

    struct Struct {
        std::string value;
    };

    TEST_CASE("BindTest, StoreByCopy") {
        Struct s = {"hello"};
        auto f = turbo::bind_front(&Struct::value, s);
        auto g = f;
        CHECK_EQ("hello", f());
        CHECK_EQ("hello", g());
        CHECK_NE(&s.value, &f());
        CHECK_NE(&s.value, &g());
        CHECK_NE(&g(), &f());
    }

    struct NonCopyable {
        explicit NonCopyable(const std::string &s) : value(s) {}

        NonCopyable(const NonCopyable &) = delete;

        NonCopyable &operator=(const NonCopyable &) = delete;

        std::string value;
    };

    const std::string &GetNonCopyableValue(const NonCopyable &n) { return n.value; }

    TEST_CASE("BindTest, StoreByRef") {
        NonCopyable s("hello");
        auto f = turbo::bind_front(&GetNonCopyableValue, std::ref(s));
        CHECK_EQ("hello", f());
        CHECK_EQ(&s.value, &f());
        auto g = std::move(f);  // NOLINT
        CHECK_EQ("hello", g());
        CHECK_EQ(&s.value, &g());
        s.value = "goodbye";
        CHECK_EQ("goodbye", g());
    }

    TEST_CASE("BindTest, StoreByCRef") {
        NonCopyable s("hello");
        auto f = turbo::bind_front(&GetNonCopyableValue, std::cref(s));
        CHECK_EQ("hello", f());
        CHECK_EQ(&s.value, &f());
        auto g = std::move(f);  // NOLINT
        CHECK_EQ("hello", g());
        CHECK_EQ(&s.value, &g());
        s.value = "goodbye";
        CHECK_EQ("goodbye", g());
    }

    const std::string &GetNonCopyableValueByWrapper(
            std::reference_wrapper<NonCopyable> n) {
        return n.get().value;
    }

    TEST_CASE("BindTest, StoreByRefInvokeByWrapper") {
        NonCopyable s("hello");
        auto f = turbo::bind_front(GetNonCopyableValueByWrapper, std::ref(s));
        CHECK_EQ("hello", f());
        CHECK_EQ(&s.value, &f());
        auto g = std::move(f);
        CHECK_EQ("hello", g());
        CHECK_EQ(&s.value, &g());
        s.value = "goodbye";
        CHECK_EQ("goodbye", g());
    }

    TEST_CASE("BindTest, StoreByPointer") {
        NonCopyable s("hello");
        auto f = turbo::bind_front(&NonCopyable::value, &s);
        CHECK_EQ("hello", f());
        CHECK_EQ(&s.value, &f());
        auto g = std::move(f);
        CHECK_EQ("hello", g());
        CHECK_EQ(&s.value, &g());
    }

    int Sink(std::unique_ptr<int> p) {
        return *p;
    }

    std::unique_ptr<int> Factory(int n) { return std::make_unique<int>(n); }

    TEST_CASE("BindTest, NonCopyableArg") {
        CHECK_EQ(42, turbo::bind_front(Sink)(std::make_unique<int>(42)));
        CHECK_EQ(42, turbo::bind_front(Sink, std::make_unique<int>(42))());
    }
/*
    TEST_CASE("BindTest, NonCopyableResult") {
        EXPECT_THAT(turbo::bind_front(Factory)(42), ::testing::Pointee(42));
        EXPECT_THAT(turbo::bind_front(Factory, 42)(), ::testing::Pointee(42));
    }
*/
// is_copy_constructible<FalseCopyable<unique_ptr<T>> is true but an attempt to
// instantiate the copy constructor leads to a compile error. This is similar
// to how standard containers behave.
    template<class T>
    struct FalseCopyable {
        FalseCopyable() {}

        FalseCopyable(const FalseCopyable &other) : m(other.m) {}

        FalseCopyable(FalseCopyable &&other) : m(std::move(other.m)) {}

        T m;
    };

    int GetMember(FalseCopyable<std::unique_ptr<int>> x) { return *x.m; }

    TEST_CASE("BindTest, WrappedMoveOnly") {
        FalseCopyable<std::unique_ptr<int>> x;
        x.m = std::make_unique<int>(42);
        auto f = turbo::bind_front(&GetMember, std::move(x));
        CHECK_EQ(42, std::move(f)());
    }

    int Plus(int a, int b) { return a + b; }

    TEST_CASE("BindTest, ConstExpr") {
        constexpr auto f = turbo::bind_front(CharAt);
        CHECK_EQ(f("ABC", 1), 'B');
        static constexpr int five = 5;
        constexpr auto plus5 = turbo::bind_front(Plus, five);
        CHECK_EQ(plus5(1), 6);

        // There seems to be a bug in MSVC dealing constexpr construction of
        // char[]. Notice 'plus5' above; 'int' works just fine.
#if !(defined(_MSC_VER) && _MSC_VER < 1910)
        static constexpr char data[] = "DEF";
        constexpr auto g = turbo::bind_front(CharAt, data);
        CHECK_EQ(g(1), 'E');
#endif
    }

    struct ManglingCall {
        int operator()(int, double, std::string) const { return 0; }
    };

    TEST_CASE("BindTest, Mangling") {
        // We just want to generate a particular instantiation to see its mangling.
        turbo::bind_front(ManglingCall{}, 1, 3.3)("A");
    }

}  // namespace
