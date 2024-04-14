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

// Tests for pointer utilities.

#include "turbo/memory/memory.h"

#include <sys/types.h>

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"


namespace {


    // This class creates observable behavior to verify that a destructor has
    // been called, via the instance_count variable.
    class DestructorVerifier {
    public:
        DestructorVerifier() { ++instance_count_; }

        DestructorVerifier(const DestructorVerifier &) = delete;

        DestructorVerifier &operator=(const DestructorVerifier &) = delete;

        ~DestructorVerifier() { --instance_count_; }

        // The number of instances of this class currently active.
        static int instance_count() { return instance_count_; }

    private:
        // The number of instances of this class currently active.
        static int instance_count_;
    };

    int DestructorVerifier::instance_count_ = 0;

    TEST_CASE("WrapUniqueTest - WrapUnique") {
        // Test that the unique_ptr is constructed properly by verifying that the
        // destructor for its payload gets called at the proper time.
        {
            auto dv = new DestructorVerifier;
            CHECK_EQ(1, DestructorVerifier::instance_count());
            std::unique_ptr<DestructorVerifier> ptr = turbo::WrapUnique(dv);
            CHECK_EQ(1, DestructorVerifier::instance_count());
        }
        CHECK_EQ(0, DestructorVerifier::instance_count());
    }

    // InitializationVerifier fills in a pattern when allocated so we can
    // distinguish between its default and value initialized states (without
    // accessing truly uninitialized memory).
    struct InitializationVerifier {
        static constexpr int kDefaultScalar = 0x43;
        static constexpr int kDefaultArray = 0x4B;

        static void *operator new(size_t n) {
            void *ret = ::operator new(n);
            memset(ret, kDefaultScalar, n);
            return ret;
        }

        static void *operator new[](size_t n) {
            void *ret = ::operator new[](n);
            memset(ret, kDefaultArray, n);
            return ret;
        }

        int a;
        int b;
    };

    struct ArrayWatch {
        void *operator new[](size_t n) {
            allocs().push_back(n);
            return ::operator new[](n);
        }

        void operator delete[](void *p) { return ::operator delete[](p); }

        static std::vector<size_t> &allocs() {
            static auto &v = *new std::vector<size_t>;
            return v;
        }
    };

    class IntPointerNonConstDeref {
    public:
        explicit IntPointerNonConstDeref(int *p) : p_(p) {}

        friend bool operator!=(const IntPointerNonConstDeref &a, std::nullptr_t) {
            return a.p_ != nullptr;
        }

        int &operator*() { return *p_; }

    private:
        std::unique_ptr<int> p_;
    };

    TEST_CASE("RawPtrTest") {
        SUBCASE("RawPointer") {
            int i = 5;
            CHECK_EQ(&i, turbo::RawPtr(&i));
        }

        SUBCASE("SmartPointer") {
            int *o = new int(5);
            std::unique_ptr<int> p(o);
            CHECK_EQ(o, turbo::RawPtr(p));
        }


        SUBCASE("SmartPointerNonConstDereference") {
            int *o = new int(5);
            IntPointerNonConstDeref p(o);
            CHECK_EQ(o, turbo::RawPtr(p));
        }

        SUBCASE("NullValuedRawPointer") {
            int *p = nullptr;
            CHECK_EQ(nullptr, turbo::RawPtr(p));
        }

        SUBCASE("NullValuedSmartPointer") {
            std::unique_ptr<int> p;
            CHECK_EQ(nullptr, turbo::RawPtr(p));
        }

        SUBCASE("Nullptr") {
            auto p = turbo::RawPtr(nullptr);
            CHECK((std::is_same<std::nullptr_t, decltype(p)>::value));
            CHECK_EQ(nullptr, p);
        }

        SUBCASE("Null") {
            auto p = turbo::RawPtr(nullptr);
            CHECK((std::is_same<std::nullptr_t, decltype(p)>::value));
            CHECK_EQ(nullptr, p);
        }

        SUBCASE("Zero") {
            auto p = turbo::RawPtr(nullptr);
            CHECK((std::is_same<std::nullptr_t, decltype(p)>::value));
            CHECK_EQ(nullptr, p);
        }
    }

    TEST_CASE("ShareUniquePtrTest") {
        SUBCASE("Share") {
            auto up = std::make_unique<int>();
            int *rp = up.get();
            auto sp = turbo::ShareUniquePtr(std::move(up));
            CHECK_EQ(sp.get(), rp);
        }

        SUBCASE("ShareNull") {
            struct NeverDie {
                using pointer = void *;

                void operator()(pointer) {
                    REQUIRE(false);
                }
            };

            std::unique_ptr<void, NeverDie> up;
            auto sp = turbo::ShareUniquePtr(std::move(up));
        }
    }

    TEST_CASE("WeakenPtrTest - Weak") {
        auto sp = std::make_shared<int>();
        auto wp = turbo::WeakenPtr(sp);
        CHECK_EQ(sp.get(), wp.lock().get());
        sp.

                reset();
        CHECK(wp.expired());
    }

// Should not compile.
/*
SUBCASE("NotAPointer) {
  turbo::RawPtr(1.5);
}
*/
    TEST_CASE("AllocatorNoThrowTest") {
        SUBCASE("DefaultAllocator") {
#if defined(TURBO_ALLOCATOR_NOTHROW) && TURBO_ALLOCATOR_NOTHROW
            CHECK(turbo::default_allocator_is_nothrow::value);
#else
            CHECK_FALSE(turbo::default_allocator_is_nothrow::value);
#endif
        }

        SUBCASE("StdAllocator") {
#if defined(TURBO_ALLOCATOR_NOTHROW) && TURBO_ALLOCATOR_NOTHROW
            CHECK(turbo::allocator_is_nothrow<std::allocator<int>>::value);
#else
            CHECK_FALSE(turbo::allocator_is_nothrow<std::allocator<int>>::value);
#endif
        }

        SUBCASE("CustomAllocator") {
            struct NoThrowAllocator {
                using is_nothrow = std::true_type;
            };
            struct CanThrowAllocator {
                using is_nothrow = std::false_type;
            };
            struct UnspecifiedAllocator {
            };
            CHECK(turbo::allocator_is_nothrow<NoThrowAllocator>::value);
            CHECK_FALSE(turbo::allocator_is_nothrow<CanThrowAllocator>::value);
            CHECK_FALSE(turbo::allocator_is_nothrow<UnspecifiedAllocator>::value);
        }
    }
}  // namespace
