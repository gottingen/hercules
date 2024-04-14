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

#include "turbo/platform/config/attribute_optimization.h"
#include <optional>
#include "turbo/platform/port.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

namespace {

// Tests for the TURBO_LIKELY and TURBO_UNLIKELY macros.
// The tests only verify that the macros are functionally correct - i.e. code
// behaves as if they weren't used. They don't try to check their impact on
// optimization.

    TEST_CASE("PredictTest, PredictTrue") {
        CHECK(TURBO_LIKELY(true));
        CHECK_FALSE(TURBO_LIKELY(false));
        CHECK(TURBO_LIKELY(1 == 1));
        CHECK_FALSE(TURBO_LIKELY(1 == 2));

        if (TURBO_LIKELY(false)) FAIL("f");
        if (!TURBO_LIKELY(true)) FAIL("FFF");

        CHECK((TURBO_LIKELY(true) && true));
        CHECK((TURBO_LIKELY(true) || false));
    }

    TEST_CASE("PredictTest, PredictFalse") {
        CHECK(TURBO_UNLIKELY(true));
        CHECK_FALSE(TURBO_UNLIKELY(false));
        CHECK(TURBO_UNLIKELY(1 == 1));
        CHECK_FALSE(TURBO_UNLIKELY(1 == 2));

        if (TURBO_UNLIKELY(false)) FAIL("f");
        if (!TURBO_UNLIKELY(true)) FAIL("f");

        CHECK((TURBO_UNLIKELY(true) && true));
        CHECK((TURBO_UNLIKELY(true) || false));
    }

    TEST_CASE("PredictTest, OneEvaluation") {
        // Verify that the expression is only evaluated once.
        int x = 0;
        if (TURBO_LIKELY((++x) == 0)) FAIL("f");
        CHECK_EQ(x, 1);
        if (TURBO_UNLIKELY((++x) == 0)) FAIL("f");
        CHECK_EQ(x, 2);
    }

    TEST_CASE("PredictTest, OperatorOrder") {
        // Verify that operator order inside and outside the macro behaves well.
        // These would fail for a naive '#define TURBO_LIKELY(x) x'
        CHECK(TURBO_LIKELY(1 && 2) == true);
        CHECK(TURBO_UNLIKELY(1 && 2) == true);
        CHECK(!TURBO_LIKELY(1 == 2));
        CHECK(!TURBO_UNLIKELY(1 == 2));
    }

    TEST_CASE("PredictTest, Pointer") {
        const int x = 3;
        const int *good_intptr = &x;
        const int *null_intptr = nullptr;
        CHECK(TURBO_LIKELY(good_intptr));
        CHECK_FALSE(TURBO_LIKELY(null_intptr));
        CHECK(TURBO_UNLIKELY(good_intptr));
        CHECK_FALSE(TURBO_UNLIKELY(null_intptr));
    }

    TEST_CASE("PredictTest, Optional") {
        // Note: An optional's truth value is the value's existence, not its truth.
        std::optional<bool> has_value(false);
        std::optional<bool> no_value;
        CHECK(TURBO_LIKELY(has_value));
        CHECK_FALSE(TURBO_LIKELY(no_value));
        CHECK(TURBO_UNLIKELY(has_value));
        CHECK_FALSE(TURBO_UNLIKELY(no_value));
    }

    class ImplictlyConvertibleToBool {
    public:
        explicit ImplictlyConvertibleToBool(bool value) : value_(value) {}

        operator bool() const {  // NOLINT(google-explicit-constructor)
            return value_;
        }

    private:
        bool value_;
    };

    TEST_CASE("PredictTest, ImplicitBoolConversion") {
        const ImplictlyConvertibleToBool is_true(true);
        const ImplictlyConvertibleToBool is_false(false);
        if (!TURBO_LIKELY(is_true))

            FAIL("f");

        if (TURBO_LIKELY(is_false))

            FAIL("f");

        if (!TURBO_UNLIKELY(is_true))

            FAIL("f");

        if (TURBO_UNLIKELY(is_false))

            FAIL("f");
    }

    class ExplictlyConvertibleToBool {
    public:
        explicit ExplictlyConvertibleToBool(bool value) : value_(value) {}

        explicit operator bool() const { return value_; }

    private:
        bool value_;
    };

    TEST_CASE("PredictTest, ExplicitBoolConversion") {
        const ExplictlyConvertibleToBool is_true(true);
        const

        ExplictlyConvertibleToBool is_false(false);
        if (!TURBO_LIKELY(is_true))
            FAIL("f");

        if (TURBO_LIKELY(is_false))
            FAIL("f");

        if (!TURBO_UNLIKELY(is_true))
            FAIL("f");

        if (TURBO_UNLIKELY(is_false))
            FAIL("f");

    }

}  // namespace
