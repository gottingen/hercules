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

#include "turbo/base/internal/throw_delegate.h"

#include <functional>
#include <new>
#include <stdexcept>

#include "turbo/platform/port.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

namespace {

    using turbo::base_internal::ThrowStdLogicError;
    using turbo::base_internal::ThrowStdInvalidArgument;
    using turbo::base_internal::ThrowStdDomainError;
    using turbo::base_internal::ThrowStdLengthError;
    using turbo::base_internal::ThrowStdOutOfRange;
    using turbo::base_internal::ThrowStdRuntimeError;
    using turbo::base_internal::ThrowStdRangeError;
    using turbo::base_internal::ThrowStdOverflowError;
    using turbo::base_internal::ThrowStdUnderflowError;
    using turbo::base_internal::ThrowStdBadFunctionCall;
    using turbo::base_internal::ThrowStdBadAlloc;

    constexpr const char *what_arg = "The quick brown fox jumps over the lazy dog";

    template<typename E>
    void ExpectThrowChar(void (*f)(const char *)) {
#ifdef TURBO_HAVE_EXCEPTIONS
        try {
            f(what_arg);
            FAIL("Didn't throw");
        } catch (const E &e) {
            REQUIRE_EQ(std::string(e.what()), std::string(what_arg));
        }
#else
        EXPECT_DEATH_IF_SUPPORTED(f(what_arg), what_arg);
#endif
    }

    template<typename E>
    void ExpectThrowString(void (*f)(const std::string &)) {
#ifdef TURBO_HAVE_EXCEPTIONS
        try {
            f(what_arg);
            FAIL("Didn't throw");
        } catch (const E &e) {
            REQUIRE_EQ(std::string(e.what()), std::string(what_arg));
        }
#else
        EXPECT_DEATH_IF_SUPPORTED(f(what_arg), what_arg);
#endif
    }

    template<typename E>
    void ExpectThrowNoWhat(void (*f)()) {
#ifdef TURBO_HAVE_EXCEPTIONS
        try {
            f();
            FAIL("Didn't throw");
        } catch (const E &e) {
        }
#else
        EXPECT_DEATH_IF_SUPPORTED(f(), "");
#endif
    }

    TEST_CASE("ThrowDelegate, ThrowStdLogicErrorChar") {
        ExpectThrowChar<std::logic_error>(ThrowStdLogicError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdInvalidArgumentChar") {
        ExpectThrowChar<std::invalid_argument>(ThrowStdInvalidArgument);
    }

    TEST_CASE("ThrowDelegate, ThrowStdDomainErrorChar") {
        ExpectThrowChar<std::domain_error>(ThrowStdDomainError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdLengthErrorChar") {
        ExpectThrowChar<std::length_error>(ThrowStdLengthError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdOutOfRangeChar") {
        ExpectThrowChar<std::out_of_range>(ThrowStdOutOfRange);
    }

    TEST_CASE("ThrowDelegate, ThrowStdRuntimeErrorChar") {
        ExpectThrowChar<std::runtime_error>(ThrowStdRuntimeError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdRangeErrorChar") {
        ExpectThrowChar<std::range_error>(ThrowStdRangeError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdOverflowErrorChar") {
        ExpectThrowChar<std::overflow_error>(ThrowStdOverflowError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdUnderflowErrorChar") {
        ExpectThrowChar<std::underflow_error>(ThrowStdUnderflowError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdLogicErrorString") {
        ExpectThrowString<std::logic_error>(ThrowStdLogicError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdInvalidArgumentString") {
        ExpectThrowString<std::invalid_argument>(ThrowStdInvalidArgument);
    }

    TEST_CASE("ThrowDelegate, ThrowStdDomainErrorString") {
        ExpectThrowString<std::domain_error>(ThrowStdDomainError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdLengthErrorString") {
        ExpectThrowString<std::length_error>(ThrowStdLengthError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdOutOfRangeString") {
        ExpectThrowString<std::out_of_range>(ThrowStdOutOfRange);
    }

    TEST_CASE("ThrowDelegate, ThrowStdRuntimeErrorString") {
        ExpectThrowString<std::runtime_error>(ThrowStdRuntimeError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdRangeErrorString") {
        ExpectThrowString<std::range_error>(ThrowStdRangeError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdOverflowErrorString") {
        ExpectThrowString<std::overflow_error>(ThrowStdOverflowError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdUnderflowErrorString") {
        ExpectThrowString<std::underflow_error>(ThrowStdUnderflowError);
    }

    TEST_CASE("ThrowDelegate, ThrowStdBadFunctionCallNoWhat") {
#ifdef TURBO_HAVE_EXCEPTIONS
        try {
            ThrowStdBadFunctionCall();
            FAIL("Didn't throw");
        } catch (const std::bad_function_call &) {
        }
#ifdef _LIBCPP_VERSION
        catch (const std::exception&) {
          // https://reviews.llvm.org/D92397 causes issues with the vtable for
          // std::bad_function_call when using libc++ as a shared library.
        }
#endif
#else
        EXPECT_DEATH_IF_SUPPORTED(ThrowStdBadFunctionCall(), "");
#endif
    }

    TEST_CASE("ThrowDelegate, ThrowStdBadAllocNoWhat") {
        ExpectThrowNoWhat<std::bad_alloc>(ThrowStdBadAlloc);
    }

}  // namespace
