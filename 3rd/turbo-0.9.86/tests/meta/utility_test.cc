// Copyright 2022 The Turbo Authors.
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

#include "turbo/meta/utility.h"

#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "turbo/memory/memory.h"
#include "turbo/platform/port.h"
#include "turbo/format/format.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

namespace {

#ifdef _MSC_VER
    // Warnings for unused variables in this test are false positives.  On other
    // platforms, they are suppressed by TURBO_MAYBE_UNUSED, but that doesn't
    // work on MSVC.
    // Both the unused variables and the name length warnings are due to calls
    // to turbo::make_index_sequence with very large values, creating very long type
    // names. The resulting warnings are so long they make build output unreadable.
#pragma warning(push)
#pragma warning(disable : 4503)  // decorated name length exceeded
#pragma warning(disable : 4101)  // unreferenced local variable
#endif                           // _MSC_VER

    template<typename F, typename Tup, size_t... Is>
    auto ApplyFromTupleImpl(F f, const Tup &tup, std::index_sequence<Is...>)
    -> decltype(f(std::get<Is>(tup)...)) {
        return f(std::get<Is>(tup)...);
    }

    template<typename Tup>
    using TupIdxSeq = std::make_index_sequence<std::tuple_size<Tup>::value>;

    template<typename F, typename Tup>
    auto ApplyFromTuple(F f, const Tup &tup)
    -> decltype(ApplyFromTupleImpl(f, tup, TupIdxSeq<Tup>{})) {
        return ApplyFromTupleImpl(f, tup, TupIdxSeq<Tup>{});
    }

    template<typename T>
    std::string Fmt(const T &x) {
        std::ostringstream os;
        os << x;
        return os.str();
    }

    struct PoorStrCat {
        template<typename... Args>
        std::string operator()(const Args &... args) const {
            std::string r;
            for (const auto &e: {Fmt(args)...}) r += e;
            return r;
        }
    };

    template<typename Tup, size_t... Is>
    std::vector<std::string> TupStringVecImpl(const Tup &tup,
                                              std::index_sequence<Is...>) {
        return {Fmt(std::get<Is>(tup))...};
    }

    template<typename... Ts>
    std::vector<std::string> TupStringVec(const std::tuple<Ts...> &tup) {
        return TupStringVecImpl(tup, std::index_sequence_for<Ts...>());
    }

    TEST_CASE("MakeIndexSequenceTest, ApplyFromTupleExample") {
        PoorStrCat f{};
        REQUIRE_EQ("12abc3.14", f(12, "abc", 3.14));
        REQUIRE_EQ("12abc3.14", ApplyFromTuple(f, std::make_tuple(12, "abc", 3.14)));
    }

    int Function(int a, int b) { return a - b; }

    int Sink(std::unique_ptr<int> p) { return *p; }

    std::unique_ptr<int> Factory(int n) { return std::make_unique<int>(n); }

    void NoOp() {}

    struct ConstFunctor {
        int operator()(int a, int b) const { return a - b; }
    };

    struct MutableFunctor {
        int operator()(int a, int b) { return a - b; }
    };

    struct EphemeralFunctor {
        EphemeralFunctor() {}

        EphemeralFunctor(const EphemeralFunctor &) {}

        EphemeralFunctor(EphemeralFunctor &&) {}

        int operator()(int a, int b) &&{ return a - b; }
    };

    struct OverloadedFunctor {
        OverloadedFunctor() {}

        OverloadedFunctor(const OverloadedFunctor &) {}

        OverloadedFunctor(OverloadedFunctor &&) {}

        template<typename... Args>
        std::string operator()(std::string_view fmt, const Args &... args) &{
            return "&" + turbo::format(fmt, args...);
        }

        template<typename... Args>
        std::string operator()(std::string_view fmt, const Args &... args) const &{
            return "const&" + turbo::format(fmt, args...);
        }

        template<typename... Args>
        std::string operator()(std::string_view fmt, const Args &... args) &&{
            return "&&" + turbo::format(fmt, args...);
        }
    };

    struct Class {
        int Method(int a, int b) { return a - b; }

        int ConstMethod(int a, int b) const { return a - b; }

        int member;
    };

    struct FlipFlop {
        int ConstMethod() const { return member; }

        FlipFlop operator*() const { return {-member}; }

        int member;
    };

    TEST_CASE("ApplyTest, Function") {
        REQUIRE_EQ(1, turbo::apply(Function, std::make_tuple(3, 2)));
        REQUIRE_EQ(1, turbo::apply(&Function, std::make_tuple(3, 2)));
    }

    TEST_CASE("ApplyTest, NonCopyableArgument") {
        REQUIRE_EQ(42, turbo::apply(Sink, std::make_tuple(std::make_unique<int>(42))));
    }


    TEST_CASE("ApplyTest, VoidResult") { turbo::apply(NoOp, std::tuple<>()); }

    TEST_CASE("ApplyTest, ConstFunctor") {
        REQUIRE_EQ(1, turbo::apply(ConstFunctor(), std::make_tuple(3, 2)));
    }

    TEST_CASE("ApplyTest, MutableFunctor") {
        MutableFunctor f;
        REQUIRE_EQ(1, turbo::apply(f, std::make_tuple(3, 2)));
        REQUIRE_EQ(1, turbo::apply(MutableFunctor(), std::make_tuple(3, 2)));
    }

    TEST_CASE("ApplyTest, EphemeralFunctor") {
        EphemeralFunctor f;
        REQUIRE_EQ(1, turbo::apply(std::move(f), std::make_tuple(3, 2)));
        REQUIRE_EQ(1, turbo::apply(EphemeralFunctor(), std::make_tuple(3, 2)));
    }

    TEST_CASE("ApplyTest, OverloadedFunctor") {
        OverloadedFunctor f;
        const OverloadedFunctor &cf = f;

        REQUIRE_EQ("&", turbo::apply(f, std::make_tuple("")));
        REQUIRE_EQ("& 42", turbo::apply(f, std::make_tuple("{}", " 42")));

        REQUIRE_EQ("const&", turbo::apply(cf, std::tuple<std::string_view>{""}));
        REQUIRE_EQ("const& 42", turbo::apply(cf, std::make_tuple("{}", " 42")));

        REQUIRE_EQ("&&", turbo::apply(std::move(f), std::tuple<std::string_view>{""}));
        OverloadedFunctor f2;
        REQUIRE_EQ("&& 42", turbo::apply(std::move(f2), std::make_tuple("{}", " 42")));
    }

    TEST_CASE("ApplyTest, ReferenceWrapper") {
        ConstFunctor cf;
        MutableFunctor mf;
        REQUIRE_EQ(1, turbo::apply(std::cref(cf), std::make_tuple(3, 2)));
        REQUIRE_EQ(1, turbo::apply(std::ref(cf), std::make_tuple(3, 2)));
        REQUIRE_EQ(1, turbo::apply(std::ref(mf), std::make_tuple(3, 2)));
    }

    TEST_CASE("ApplyTest, MemberFunction") {
        std::unique_ptr<Class> p(new Class);
        std::unique_ptr<const Class> cp(new Class);
        REQUIRE_EQ(
                1, turbo::apply(&Class::Method,
                                std::tuple<std::unique_ptr<Class> &, int, int>(p, 3, 2)));
        REQUIRE_EQ(1, turbo::apply(&Class::Method,
                                   std::tuple<Class *, int, int>(p.get(), 3, 2)));
        REQUIRE_EQ(
                1, turbo::apply(&Class::Method, std::tuple<Class &, int, int>(*p, 3, 2)));

        REQUIRE_EQ(
                1, turbo::apply(&Class::ConstMethod,
                                std::tuple<std::unique_ptr<Class> &, int, int>(p, 3, 2)));
        REQUIRE_EQ(1, turbo::apply(&Class::ConstMethod,
                                   std::tuple<Class *, int, int>(p.get(), 3, 2)));
        REQUIRE_EQ(1, turbo::apply(&Class::ConstMethod,
                                   std::tuple<Class &, int, int>(*p, 3, 2)));

        REQUIRE_EQ(1, turbo::apply(&Class::ConstMethod,
                                   std::tuple<std::unique_ptr<const Class> &, int, int>(
                                           cp, 3, 2)));
        REQUIRE_EQ(1, turbo::apply(&Class::ConstMethod,
                                   std::tuple<const Class *, int, int>(cp.get(), 3, 2)));
        REQUIRE_EQ(1, turbo::apply(&Class::ConstMethod,
                                   std::tuple<const Class &, int, int>(*cp, 3, 2)));

        REQUIRE_EQ(1, turbo::apply(&Class::Method,
                                   std::make_tuple(std::make_unique<Class>(), 3, 2)));
        REQUIRE_EQ(1, turbo::apply(&Class::ConstMethod,
                                   std::make_tuple(std::make_unique<Class>(), 3, 2)));
        REQUIRE_EQ(
                1, turbo::apply(&Class::ConstMethod,
                                std::make_tuple(std::make_unique<const Class>(), 3, 2)));
    }

    TEST_CASE("ApplyTest, DataMember") {
        std::unique_ptr<Class> p(new Class{42});
        std::unique_ptr<const Class> cp(new Class{42});
        REQUIRE_EQ(
                42, turbo::apply(&Class::member, std::tuple<std::unique_ptr<Class> &>(p)));
        REQUIRE_EQ(42, turbo::apply(&Class::member, std::tuple<Class &>(*p)));
        REQUIRE_EQ(42, turbo::apply(&Class::member, std::tuple<Class *>(p.get())));

        turbo::apply(&Class::member, std::tuple<std::unique_ptr<Class> &>(p)) = 42;
        turbo::apply(&Class::member, std::tuple<Class *>(p.get())) = 42;
        turbo::apply(&Class::member, std::tuple<Class &>(*p)) = 42;

        REQUIRE_EQ(42, turbo::apply(&Class::member,
                                    std::tuple<std::unique_ptr<const Class> &>(cp)));
        REQUIRE_EQ(42, turbo::apply(&Class::member, std::tuple<const Class &>(*cp)));
        REQUIRE_EQ(42,
                   turbo::apply(&Class::member, std::tuple<const Class *>(cp.get())));
    }

    TEST_CASE("ApplyTest, FlipFlop") {
        FlipFlop obj = {42};
        // This call could resolve to (obj.*&FlipFlop::ConstMethod)() or
        // ((*obj).*&FlipFlop::ConstMethod)(). We verify that it's the former.
        REQUIRE_EQ(42, turbo::apply(&FlipFlop::ConstMethod, std::make_tuple(obj)));
        REQUIRE_EQ(42, turbo::apply(&FlipFlop::member, std::make_tuple(obj)));
    }

    TEST_CASE("ExchangeTest, MoveOnly") {
        auto a = Factory(1);
        REQUIRE_EQ(1, *a);
        auto b = turbo::exchange(a, Factory(2));
        REQUIRE_EQ(2, *a);
        REQUIRE_EQ(1, *b);
    }

    TEST_CASE("MakeFromTupleTest, String") {
        REQUIRE_EQ(
                turbo::make_from_tuple<std::string>(std::make_tuple("hello world", 5)),
                "hello");
    }

    TEST_CASE("MakeFromTupleTest, MoveOnlyParameter") {
        struct S {
            S(std::unique_ptr<int> n, std::unique_ptr<int> m) : value(*n + *m) {}

            int value = 0;
        };
        auto tup =
                std::make_tuple(std::make_unique<int>(3), std::make_unique<int>(4));
        auto s = turbo::make_from_tuple<S>(std::move(tup));
        REQUIRE_EQ(s.value, 7);
    }

    TEST_CASE("MakeFromTupleTest, NoParameters") {
        struct S {
            S() : value(1) {}

            int value = 2;
        };
        REQUIRE_EQ(turbo::make_from_tuple<S>(std::make_tuple()).value, 1);
    }

    TEST_CASE("MakeFromTupleTest, Pair") {
        REQUIRE_EQ(
                (turbo::make_from_tuple<std::pair<bool, int>>(std::make_tuple(true, 17))),
                std::make_pair(true, 17));
    }

}  // namespace
