// Copyright 2019 The Turbo Authors.
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

#include "turbo/meta/function_ref.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include <functional>
#include <memory>

#include "tests/container/test_instance_tracker.h"
#include "turbo/memory/memory.h"

namespace turbo {
namespace {

void RunFun(FunctionRef<void()> f) { f(); }
int Function() { return 1337; }
    int NoExceptFunction() noexcept { return 1337; }
    
TEST_CASE("FunctionRefTest") {

        SUBCASE("Lambda") {
            bool ran = false;
            RunFun([&] { ran = true; });
            REQUIRE(ran);
        }

        SUBCASE("Function1") {
            FunctionRef<int()> ref(&Function);
            REQUIRE_EQ(1337, ref());
        }

        SUBCASE("Function2") {
            FunctionRef<int()> ref(Function);
            REQUIRE_EQ(1337, ref());
        }



// TODO(jdennett): Add a test for noexcept member functions.
        SUBCASE("NoExceptFunction") {
            FunctionRef<int()> ref(NoExceptFunction);
            REQUIRE_EQ(1337, ref());
        }

        SUBCASE("ForwardsArgs") {
            auto l = [](std::unique_ptr<int> i) { return *i; };
            FunctionRef<int(std::unique_ptr<int>)> ref(l);
            REQUIRE_EQ(42, ref(std::make_unique<int>(42)));
        }
    }
TEST_CASE("FunctionRef") {
        SUBCASE("ReturnMoveOnly") {
            auto l = [] { return std::make_unique<int>(29); };
            FunctionRef<std::unique_ptr<int>()> ref(l);
            REQUIRE_EQ(29, *ref());
        }


        SUBCASE("ManyArgs") {
            auto l = [](int a, int b, int c) { return a + b + c; };
            FunctionRef<int(int, int, int)> ref(l);
            REQUIRE_EQ(6, ref(1, 2, 3));
        }

        SUBCASE("VoidResultFromNonVoidFunctor") {
            bool ran = false;
            auto l = [&]() -> int {
                ran = true;
                return 2;
            };
            FunctionRef<void()> ref(l);
            ref();
            REQUIRE(ran);
        }

        SUBCASE("CastFromDerived") {
            struct Base {
            };
            struct Derived : public Base {
            };

            Derived d;
            auto l1 = [&](Base *b) { REQUIRE_EQ(&d, b); };
            FunctionRef<void(Derived *)> ref1(l1);
            ref1(&d);

            auto l2 = [&]() -> Derived * { return &d; };
            FunctionRef<Base *()> ref2(l2);
            REQUIRE_EQ(&d, ref2());
        }

        SUBCASE("VoidResultFromNonVoidFuncton") {
            FunctionRef<void()> ref(Function);
            ref();
        }

        SUBCASE("MemberPtr") {
            struct S {
                int i;
            };

            S s{1100111};
            auto mem_ptr = &S::i;
            FunctionRef<int(const S &s)> ref(mem_ptr);
            REQUIRE_EQ(1100111, ref(s));
        }

        SUBCASE("MemberFun") {
            struct S {
                int i;

                int get_i() const { return i; }
            };

            S s{22};
            auto mem_fun_ptr = &S::get_i;
            FunctionRef<int(const S &s)> ref(mem_fun_ptr);
            REQUIRE_EQ(22, ref(s));
        }

        SUBCASE("MemberFunRefqualified") {
            struct S {
                int i;

                int get_i() &&{ return i; }
            };
            auto mem_fun_ptr = &S::get_i;
            S s{22};
            FunctionRef<int(S &&s)> ref(mem_fun_ptr);
            REQUIRE_EQ(22, ref(std::move(s)));
        }

        SUBCASE("CopiesAndMovesPerPassByValue") {
            turbo::test_internal::InstanceTracker tracker;
            turbo::test_internal::CopyableMovableInstance instance(0);
            auto l = [](turbo::test_internal::CopyableMovableInstance) {};
            FunctionRef<void(turbo::test_internal::CopyableMovableInstance)> ref(l);
            ref(instance);
            REQUIRE_EQ(tracker.copies(), 1);
            REQUIRE_EQ(tracker.moves(), 1);
        }

        SUBCASE("CopiesAndMovesPerPassByRef") {
            turbo::test_internal::InstanceTracker tracker;
            turbo::test_internal::CopyableMovableInstance instance(0);
            auto l = [](const turbo::test_internal::CopyableMovableInstance &) {};
            FunctionRef<void(const turbo::test_internal::CopyableMovableInstance &)> ref(l);
            ref(instance);
            REQUIRE_EQ(tracker.copies(), 0);
            REQUIRE_EQ(tracker.moves(), 0);
        }

        SUBCASE("CopiesAndMovesPerPassByValueCallByMove") {
            turbo::test_internal::InstanceTracker tracker;
            turbo::test_internal::CopyableMovableInstance instance(0);
            auto l = [](turbo::test_internal::CopyableMovableInstance) {};
            FunctionRef<void(turbo::test_internal::CopyableMovableInstance)> ref(l);
            ref(std::move(instance));
            REQUIRE_EQ(tracker.copies(), 0);
            REQUIRE_EQ(tracker.moves(), 2);
        }

        SUBCASE("CopiesAndMovesPerPassByValueToRef") {
            turbo::test_internal::InstanceTracker tracker;
            turbo::test_internal::CopyableMovableInstance instance(0);
            auto l = [](const turbo::test_internal::CopyableMovableInstance &) {};
            FunctionRef<void(turbo::test_internal::CopyableMovableInstance)> ref(l);
            ref(std::move(instance));
            REQUIRE_EQ(tracker.copies(), 0);
            REQUIRE_EQ(tracker.moves(), 1);
        }

        SUBCASE("PassByValueTypes") {
            using turbo::functional_internal::Invoker;
            using turbo::functional_internal::VoidPtr;
            using turbo::test_internal::CopyableMovableInstance;
            struct Trivial {
                void *p[2];
            };
            struct LargeTrivial {
                void *p[3];
            };

            static_assert(std::is_same<Invoker<void, int>, void (*)(VoidPtr, int)>::value,
                          "Scalar types should be passed by value");
            static_assert(
                    std::is_same<Invoker<void, Trivial>, void (*)(VoidPtr, Trivial)>::value,
                    "Small trivial types should be passed by value");
            static_assert(std::is_same<Invoker<void, LargeTrivial>,
                                  void (*)(VoidPtr, LargeTrivial &&)>::value,
                          "Large trivial types should be passed by rvalue reference");
            static_assert(
                    std::is_same<Invoker<void, CopyableMovableInstance>,
                            void (*)(VoidPtr, CopyableMovableInstance &&)>::value,
                    "Types with copy/move ctor should be passed by rvalue reference");

            // References are passed as references.
            static_assert(
                    std::is_same<Invoker<void, int &>, void (*)(VoidPtr, int &)>::value,
                    "Reference types should be preserved");
            static_assert(
                    std::is_same<Invoker<void, CopyableMovableInstance &>,
                            void (*)(VoidPtr, CopyableMovableInstance &)>::value,
                    "Reference types should be preserved");
            static_assert(
                    std::is_same<Invoker<void, CopyableMovableInstance &&>,
                            void (*)(VoidPtr, CopyableMovableInstance &&)>::value,
                    "Reference types should be preserved");

            // Make sure the address of an object received by reference is the same as the
            // addess of the object passed by the caller.
            {
                LargeTrivial obj;
                auto test = [&obj](LargeTrivial &input) { REQUIRE_EQ(&input, &obj); };
                turbo::FunctionRef<void(LargeTrivial &)> ref(test);
                ref(obj);
            }

            {
                Trivial obj;
                auto test = [&obj](Trivial &input) { REQUIRE_EQ(&input, &obj); };
                turbo::FunctionRef<void(Trivial &)> ref(test);
                ref(obj);
            }
        }
    }
}  // namespace
}  // namespace turbo
