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
//
// -----------------------------------------------------------------------------
// File: mock_distributions.h
// -----------------------------------------------------------------------------
//
// This file contains mock distribution functions for use alongside an
// `turbo::MockingBitGen` object within the Googletest testing framework. Such
// mocks are useful to provide deterministic values as return values within
// (otherwise random) Turbo distribution functions.
//
// The return type of each function is a mock expectation object which
// is used to set the match result.
//
// More information about the Googletest testing framework is available at
// https://github.com/google/googletest
//
// EXPECT_CALL and ON_CALL need to be made within the same DLL component as
// the call to turbo::uniform and related methods, otherwise mocking will fail
// since the  underlying implementation creates a type-specific pointer which
// will be distinct across different DLL boundaries.
//
// Example:
//
//   turbo::MockingBitGen mock;
//   EXPECT_CALL(turbo::MockUniform<int>(), Call(mock, 1, 1000))
//     .WillRepeatedly(testing::ReturnRoundRobin({20, 40}));
//
//   EXPECT_EQ(turbo::uniform<int>(gen, 1, 1000), 20);
//   EXPECT_EQ(turbo::uniform<int>(gen, 1, 1000), 40);
//   EXPECT_EQ(turbo::uniform<int>(gen, 1, 1000), 20);
//   EXPECT_EQ(turbo::uniform<int>(gen, 1, 1000), 40);

#ifndef TURBO_RANDOM_MOCK_DISTRIBUTIONS_H_
#define TURBO_RANDOM_MOCK_DISTRIBUTIONS_H_

#include <limits>
#include <type_traits>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "turbo/meta/type_traits.h"
#include "turbo/random/fwd.h"
#include "tests/random/mock_overload_set.h"
#include "tests/random/mocking_bit_gen.h"

namespace turbo {


// -----------------------------------------------------------------------------
// turbo::MockUniform
// -----------------------------------------------------------------------------
//
// Matches calls to turbo::uniform.
//
// `turbo::MockUniform` is a class template used in conjunction with Googletest's
// `ON_CALL()` and `EXPECT_CALL()` macros. To use it, default-construct an
// instance of it inside `ON_CALL()` or `EXPECT_CALL()`, and use `Call(...)` the
// same way one would define mocks on a Googletest `MockFunction()`.
//
// Example:
//
//  turbo::MockingBitGen mock;
//  EXPECT_CALL(turbo::MockUniform<uint32_t>(), Call(mock))
//     .WillOnce(Return(123456));
//  auto x = turbo::uniform<uint32_t>(mock);
//  assert(x == 123456)
//
template <typename R>
using MockUniform = random_internal::MockOverloadSet<
    random_internal::UniformDistributionWrapper<R>,
    R(IntervalClosedOpenTag, MockingBitGen&, R, R),
    R(IntervalClosedClosedTag, MockingBitGen&, R, R),
    R(IntervalOpenOpenTag, MockingBitGen&, R, R),
    R(IntervalOpenClosedTag, MockingBitGen&, R, R), R(MockingBitGen&, R, R),
    R(MockingBitGen&)>;

// -----------------------------------------------------------------------------
// turbo::MockBernoulli
// -----------------------------------------------------------------------------
//
// Matches calls to turbo::bernoulli.
//
// `turbo::MockBernoulli` is a class used in conjunction with Googletest's
// `ON_CALL()` and `EXPECT_CALL()` macros. To use it, default-construct an
// instance of it inside `ON_CALL()` or `EXPECT_CALL()`, and use `Call(...)` the
// same way one would define mocks on a Googletest `MockFunction()`.
//
// Example:
//
//  turbo::MockingBitGen mock;
//  EXPECT_CALL(turbo::MockBernoulli(), Call(mock, testing::_))
//     .WillOnce(Return(false));
//  assert(turbo::bernoulli(mock, 0.5) == false);
//
using MockBernoulli =
    random_internal::MockOverloadSet<turbo::bernoulli_distribution,
                                     bool(MockingBitGen&, double)>;

// -----------------------------------------------------------------------------
// turbo::MockBeta
// -----------------------------------------------------------------------------
//
// Matches calls to turbo::beta.
//
// `turbo::MockBeta` is a class used in conjunction with Googletest's `ON_CALL()`
// and `EXPECT_CALL()` macros. To use it, default-construct an instance of it
// inside `ON_CALL()` or `EXPECT_CALL()`, and use `Call(...)` the same way one
// would define mocks on a Googletest `MockFunction()`.
//
// Example:
//
//  turbo::MockingBitGen mock;
//  EXPECT_CALL(turbo::MockBeta(), Call(mock, 3.0, 2.0))
//     .WillOnce(Return(0.567));
//  auto x = turbo::beta<double>(mock, 3.0, 2.0);
//  assert(x == 0.567);
//
template <typename RealType>
using MockBeta =
    random_internal::MockOverloadSet<turbo::beta_distribution<RealType>,
                                     RealType(MockingBitGen&, RealType,
                                              RealType)>;

// -----------------------------------------------------------------------------
// turbo::MockExponential
// -----------------------------------------------------------------------------
//
// Matches calls to turbo::exponential.
//
// `turbo::MockExponential` is a class template used in conjunction with
// Googletest's `ON_CALL()` and `EXPECT_CALL()` macros. To use it,
// default-construct an instance of it inside `ON_CALL()` or `EXPECT_CALL()`,
// and use `Call(...)` the same way one would define mocks on a
// Googletest `MockFunction()`.
//
// Example:
//
//  turbo::MockingBitGen mock;
//  EXPECT_CALL(turbo::MockExponential<double>(), Call(mock, 0.5))
//     .WillOnce(Return(12.3456789));
//  auto x = turbo::exponential<double>(mock, 0.5);
//  assert(x == 12.3456789)
//
template <typename RealType>
using MockExponential =
    random_internal::MockOverloadSet<turbo::exponential_distribution<RealType>,
                                     RealType(MockingBitGen&, RealType)>;

// -----------------------------------------------------------------------------
// turbo::MockGaussian
// -----------------------------------------------------------------------------
//
// Matches calls to turbo::gaussian.
//
// `turbo::MockGaussian` is a class template used in conjunction with
// Googletest's `ON_CALL()` and `EXPECT_CALL()` macros. To use it,
// default-construct an instance of it inside `ON_CALL()` or `EXPECT_CALL()`,
// and use `Call(...)` the same way one would define mocks on a
// Googletest `MockFunction()`.
//
// Example:
//
//  turbo::MockingBitGen mock;
//  EXPECT_CALL(turbo::MockGaussian<double>(), Call(mock, 16.3, 3.3))
//     .WillOnce(Return(12.3456789));
//  auto x = turbo::gaussian<double>(mock, 16.3, 3.3);
//  assert(x == 12.3456789)
//
template <typename RealType>
using MockGaussian =
    random_internal::MockOverloadSet<turbo::gaussian_distribution<RealType>,
                                     RealType(MockingBitGen&, RealType,
                                              RealType)>;

// -----------------------------------------------------------------------------
// turbo::MockLogUniform
// -----------------------------------------------------------------------------
//
// Matches calls to turbo::log_uniform.
//
// `turbo::MockLogUniform` is a class template used in conjunction with
// Googletest's `ON_CALL()` and `EXPECT_CALL()` macros. To use it,
// default-construct an instance of it inside `ON_CALL()` or `EXPECT_CALL()`,
// and use `Call(...)` the same way one would define mocks on a
// Googletest `MockFunction()`.
//
// Example:
//
//  turbo::MockingBitGen mock;
//  EXPECT_CALL(turbo::MockLogUniform<int>(), Call(mock, 10, 10000, 10))
//     .WillOnce(Return(1221));
//  auto x = turbo::log_uniform<int>(mock, 10, 10000, 10);
//  assert(x == 1221)
//
template <typename IntType>
using MockLogUniform = random_internal::MockOverloadSet<
    turbo::log_uniform_int_distribution<IntType>,
    IntType(MockingBitGen&, IntType, IntType, IntType)>;

// -----------------------------------------------------------------------------
// turbo::MockPoisson
// -----------------------------------------------------------------------------
//
// Matches calls to turbo::poisson.
//
// `turbo::MockPoisson` is a class template used in conjunction with Googletest's
// `ON_CALL()` and `EXPECT_CALL()` macros. To use it, default-construct an
// instance of it inside `ON_CALL()` or `EXPECT_CALL()`, and use `Call(...)` the
// same way one would define mocks on a Googletest `MockFunction()`.
//
// Example:
//
//  turbo::MockingBitGen mock;
//  EXPECT_CALL(turbo::MockPoisson<int>(), Call(mock, 2.0))
//     .WillOnce(Return(1221));
//  auto x = turbo::poisson<int>(mock, 2.0);
//  assert(x == 1221)
//
template <typename IntType>
using MockPoisson =
    random_internal::MockOverloadSet<turbo::poisson_distribution<IntType>,
                                     IntType(MockingBitGen&, double)>;

// -----------------------------------------------------------------------------
// turbo::MockZipf
// -----------------------------------------------------------------------------
//
// Matches calls to turbo::zipf.
//
// `turbo::MockZipf` is a class template used in conjunction with Googletest's
// `ON_CALL()` and `EXPECT_CALL()` macros. To use it, default-construct an
// instance of it inside `ON_CALL()` or `EXPECT_CALL()`, and use `Call(...)` the
// same way one would define mocks on a Googletest `MockFunction()`.
//
// Example:
//
//  turbo::MockingBitGen mock;
//  EXPECT_CALL(turbo::MockZipf<int>(), Call(mock, 1000000, 2.0, 1.0))
//     .WillOnce(Return(1221));
//  auto x = turbo::zipf<int>(mock, 1000000, 2.0, 1.0);
//  assert(x == 1221)
//
template <typename IntType>
using MockZipf =
    random_internal::MockOverloadSet<turbo::zipf_distribution<IntType>,
                                     IntType(MockingBitGen&, IntType, double,
                                             double)>;


}  // namespace turbo

#endif  // TURBO_RANDOM_MOCK_DISTRIBUTIONS_H_
