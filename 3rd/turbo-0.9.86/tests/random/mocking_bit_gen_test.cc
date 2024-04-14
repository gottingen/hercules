//
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
#include "tests/random/mocking_bit_gen.h"

#include <cmath>
#include <numeric>
#include <random>

#include "gmock/gmock.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "turbo/random/bit_gen_ref.h"
#include "tests/random/mock_distributions.h"
#include "turbo/random/random.h"

namespace {

using ::testing::_;
using ::testing::Ne;
using ::testing::Return;

TEST(BasicMocking, AllDistributionsAreOverridable) {
  turbo::MockingBitGen gen;

  EXPECT_NE(turbo::uniform<int>(gen, 1, 1000000), 20);
  EXPECT_CALL(turbo::MockUniform<int>(), Call(gen, 1, 1000000))
      .WillOnce(Return(20));
  EXPECT_EQ(turbo::uniform<int>(gen, 1, 1000000), 20);

  EXPECT_NE(turbo::uniform<double>(gen, 0.0, 100.0), 5.0);
  EXPECT_CALL(turbo::MockUniform<double>(), Call(gen, 0.0, 100.0))
      .WillOnce(Return(5.0));
  EXPECT_EQ(turbo::uniform<double>(gen, 0.0, 100.0), 5.0);

  EXPECT_NE(turbo::exponential<double>(gen, 1.0), 42);
  EXPECT_CALL(turbo::MockExponential<double>(), Call(gen, 1.0))
      .WillOnce(Return(42));
  EXPECT_EQ(turbo::exponential<double>(gen, 1.0), 42);

  EXPECT_NE(turbo::poisson<int>(gen, 1.0), 500);
  EXPECT_CALL(turbo::MockPoisson<int>(), Call(gen, 1.0)).WillOnce(Return(500));
  EXPECT_EQ(turbo::poisson<int>(gen, 1.0), 500);

  EXPECT_NE(turbo::bernoulli(gen, 0.000001), true);
  EXPECT_CALL(turbo::MockBernoulli(), Call(gen, 0.000001))
      .WillOnce(Return(true));
  EXPECT_EQ(turbo::bernoulli(gen, 0.000001), true);

  EXPECT_NE(turbo::zipf<int>(gen, 1000000, 2.0, 1.0), 1221);
  EXPECT_CALL(turbo::MockZipf<int>(), Call(gen, 1000000, 2.0, 1.0))
      .WillOnce(Return(1221));
  EXPECT_EQ(turbo::zipf<int>(gen, 1000000, 2.0, 1.0), 1221);

  EXPECT_NE(turbo::gaussian<double>(gen, 0.0, 1.0), 0.001);
  EXPECT_CALL(turbo::MockGaussian<double>(), Call(gen, 0.0, 1.0))
      .WillOnce(Return(0.001));
  EXPECT_EQ(turbo::gaussian<double>(gen, 0.0, 1.0), 0.001);

  EXPECT_NE(turbo::log_uniform<int>(gen, 0, 1000000, 2), 500000);
  EXPECT_CALL(turbo::MockLogUniform<int>(), Call(gen, 0, 1000000, 2))
      .WillOnce(Return(500000));
  EXPECT_EQ(turbo::log_uniform<int>(gen, 0, 1000000, 2), 500000);
}

TEST(BasicMocking, OnDistribution) {
  turbo::MockingBitGen gen;

  EXPECT_NE(turbo::uniform<int>(gen, 1, 1000000), 20);
  ON_CALL(turbo::MockUniform<int>(), Call(gen, 1, 1000000))
      .WillByDefault(Return(20));
  EXPECT_EQ(turbo::uniform<int>(gen, 1, 1000000), 20);

  EXPECT_NE(turbo::uniform<double>(gen, 0.0, 100.0), 5.0);
  ON_CALL(turbo::MockUniform<double>(), Call(gen, 0.0, 100.0))
      .WillByDefault(Return(5.0));
  EXPECT_EQ(turbo::uniform<double>(gen, 0.0, 100.0), 5.0);

  EXPECT_NE(turbo::exponential<double>(gen, 1.0), 42);
  ON_CALL(turbo::MockExponential<double>(), Call(gen, 1.0))
      .WillByDefault(Return(42));
  EXPECT_EQ(turbo::exponential<double>(gen, 1.0), 42);

  EXPECT_NE(turbo::poisson<int>(gen, 1.0), 500);
  ON_CALL(turbo::MockPoisson<int>(), Call(gen, 1.0)).WillByDefault(Return(500));
  EXPECT_EQ(turbo::poisson<int>(gen, 1.0), 500);

  EXPECT_NE(turbo::bernoulli(gen, 0.000001), true);
  ON_CALL(turbo::MockBernoulli(), Call(gen, 0.000001))
      .WillByDefault(Return(true));
  EXPECT_EQ(turbo::bernoulli(gen, 0.000001), true);

  EXPECT_NE(turbo::zipf<int>(gen, 1000000, 2.0, 1.0), 1221);
  ON_CALL(turbo::MockZipf<int>(), Call(gen, 1000000, 2.0, 1.0))
      .WillByDefault(Return(1221));
  EXPECT_EQ(turbo::zipf<int>(gen, 1000000, 2.0, 1.0), 1221);

  EXPECT_NE(turbo::gaussian<double>(gen, 0.0, 1.0), 0.001);
  ON_CALL(turbo::MockGaussian<double>(), Call(gen, 0.0, 1.0))
      .WillByDefault(Return(0.001));
  EXPECT_EQ(turbo::gaussian<double>(gen, 0.0, 1.0), 0.001);

  EXPECT_NE(turbo::log_uniform<int>(gen, 0, 1000000, 2), 2040);
  ON_CALL(turbo::MockLogUniform<int>(), Call(gen, 0, 1000000, 2))
      .WillByDefault(Return(2040));
  EXPECT_EQ(turbo::log_uniform<int>(gen, 0, 1000000, 2), 2040);
}

TEST(BasicMocking, GMockMatchers) {
  turbo::MockingBitGen gen;

  EXPECT_NE(turbo::zipf<int>(gen, 1000000, 2.0, 1.0), 1221);
  ON_CALL(turbo::MockZipf<int>(), Call(gen, 1000000, 2.0, 1.0))
      .WillByDefault(Return(1221));
  EXPECT_EQ(turbo::zipf<int>(gen, 1000000, 2.0, 1.0), 1221);
}

TEST(BasicMocking, OverridesWithMultipleGMockExpectations) {
  turbo::MockingBitGen gen;

  EXPECT_CALL(turbo::MockUniform<int>(), Call(gen, 1, 10000))
      .WillOnce(Return(20))
      .WillOnce(Return(40))
      .WillOnce(Return(60));
  EXPECT_EQ(turbo::uniform(gen, 1, 10000), 20);
  EXPECT_EQ(turbo::uniform(gen, 1, 10000), 40);
  EXPECT_EQ(turbo::uniform(gen, 1, 10000), 60);
}

TEST(BasicMocking, DefaultArgument) {
  turbo::MockingBitGen gen;

  ON_CALL(turbo::MockExponential<double>(), Call(gen, 1.0))
      .WillByDefault(Return(200));

  EXPECT_EQ(turbo::exponential<double>(gen), 200);
  EXPECT_EQ(turbo::exponential<double>(gen, 1.0), 200);
}

TEST(BasicMocking, MultipleGenerators) {
  auto get_value = [](turbo::BitGenRef gen_ref) {
    return turbo::uniform(gen_ref, 1, 1000000);
  };
  turbo::MockingBitGen unmocked_generator;
  turbo::MockingBitGen mocked_with_3;
  turbo::MockingBitGen mocked_with_11;

  EXPECT_CALL(turbo::MockUniform<int>(), Call(mocked_with_3, 1, 1000000))
      .WillOnce(Return(3))
      .WillRepeatedly(Return(17));
  EXPECT_CALL(turbo::MockUniform<int>(), Call(mocked_with_11, 1, 1000000))
      .WillOnce(Return(11))
      .WillRepeatedly(Return(17));

  // Ensure that unmocked generator generates neither value.
  int unmocked_value = get_value(unmocked_generator);
  EXPECT_NE(unmocked_value, 3);
  EXPECT_NE(unmocked_value, 11);
  // Mocked generators should generate their mocked values.
  EXPECT_EQ(get_value(mocked_with_3), 3);
  EXPECT_EQ(get_value(mocked_with_11), 11);
  // Ensure that the mocks have expired.
  EXPECT_NE(get_value(mocked_with_3), 3);
  EXPECT_NE(get_value(mocked_with_11), 11);
}

TEST(BasicMocking, MocksNotTrigeredForIncorrectTypes) {
  turbo::MockingBitGen gen;
  EXPECT_CALL(turbo::MockUniform<uint32_t>(), Call(gen)).WillOnce(Return(42));

  EXPECT_NE(turbo::uniform<uint16_t>(gen), 42);  // Not mocked
  EXPECT_EQ(turbo::uniform<uint32_t>(gen), 42);  // Mock triggered
}

TEST(BasicMocking, FailsOnUnsatisfiedMocks) {
  EXPECT_NONFATAL_FAILURE(
      []() {
        turbo::MockingBitGen gen;
        EXPECT_CALL(turbo::MockExponential<double>(), Call(gen, 1.0))
            .WillOnce(Return(3.0));
        // Does not call turbo::exponential().
      }(),
      "unsatisfied and active");
}

TEST(OnUniform, RespectsUniformIntervalSemantics) {
  turbo::MockingBitGen gen;

  EXPECT_CALL(turbo::MockUniform<int>(),
              Call(turbo::IntervalClosed, gen, 1, 1000000))
      .WillOnce(Return(301));
  EXPECT_NE(turbo::uniform(gen, 1, 1000000), 301);  // Not mocked
  EXPECT_EQ(turbo::uniform(turbo::IntervalClosed, gen, 1, 1000000), 301);
}

TEST(OnUniform, RespectsNoArgUnsignedShorthand) {
  turbo::MockingBitGen gen;
  EXPECT_CALL(turbo::MockUniform<uint32_t>(), Call(gen)).WillOnce(Return(42));
  EXPECT_EQ(turbo::uniform<uint32_t>(gen), 42);
}

TEST(RepeatedlyModifier, ForceSnakeEyesForManyDice) {
  auto roll_some_dice = [](turbo::BitGenRef gen_ref) {
    std::vector<int> results(16);
    for (auto& r : results) {
      r = turbo::uniform(turbo::IntervalClosed, gen_ref, 1, 6);
    }
    return results;
  };
  std::vector<int> results;
  turbo::MockingBitGen gen;

  // Without any mocked calls, not all dice roll a "6".
  results = roll_some_dice(gen);
  EXPECT_LT(std::accumulate(std::begin(results), std::end(results), 0),
            results.size() * 6);

  // Verify that we can force all "6"-rolls, with mocking.
  ON_CALL(turbo::MockUniform<int>(), Call(turbo::IntervalClosed, gen, 1, 6))
      .WillByDefault(Return(6));
  results = roll_some_dice(gen);
  EXPECT_EQ(std::accumulate(std::begin(results), std::end(results), 0),
            results.size() * 6);
}

TEST(WillOnce, DistinctCounters) {
  turbo::MockingBitGen gen;
  EXPECT_CALL(turbo::MockUniform<int>(), Call(gen, 1, 1000000))
      .Times(3)
      .WillRepeatedly(Return(0));
  EXPECT_CALL(turbo::MockUniform<int>(), Call(gen, 1000001, 2000000))
      .Times(3)
      .WillRepeatedly(Return(1));
  EXPECT_EQ(turbo::uniform(gen, 1000001, 2000000), 1);
  EXPECT_EQ(turbo::uniform(gen, 1, 1000000), 0);
  EXPECT_EQ(turbo::uniform(gen, 1000001, 2000000), 1);
  EXPECT_EQ(turbo::uniform(gen, 1, 1000000), 0);
  EXPECT_EQ(turbo::uniform(gen, 1000001, 2000000), 1);
  EXPECT_EQ(turbo::uniform(gen, 1, 1000000), 0);
}

TEST(TimesModifier, ModifierSaturatesAndExpires) {
  EXPECT_NONFATAL_FAILURE(
      []() {
        turbo::MockingBitGen gen;
        EXPECT_CALL(turbo::MockUniform<int>(), Call(gen, 1, 1000000))
            .Times(3)
            .WillRepeatedly(Return(15))
            .RetiresOnSaturation();

        EXPECT_EQ(turbo::uniform(gen, 1, 1000000), 15);
        EXPECT_EQ(turbo::uniform(gen, 1, 1000000), 15);
        EXPECT_EQ(turbo::uniform(gen, 1, 1000000), 15);
        // Times(3) has expired - Should get a different value now.

        EXPECT_NE(turbo::uniform(gen, 1, 1000000), 15);
      }(),
      "");
}

TEST(TimesModifier, Times0) {
  turbo::MockingBitGen gen;
  EXPECT_CALL(turbo::MockBernoulli(), Call(gen, 0.0)).Times(0);
  EXPECT_CALL(turbo::MockPoisson<int>(), Call(gen, 1.0)).Times(0);
}

TEST(AnythingMatcher, MatchesAnyArgument) {
  using testing::_;

  {
    turbo::MockingBitGen gen;
    ON_CALL(turbo::MockUniform<int>(), Call(turbo::IntervalClosed, gen, _, 1000))
        .WillByDefault(Return(11));
    ON_CALL(turbo::MockUniform<int>(),
            Call(turbo::IntervalClosed, gen, _, Ne(1000)))
        .WillByDefault(Return(99));

    EXPECT_EQ(turbo::uniform(turbo::IntervalClosed, gen, 10, 1000000), 99);
    EXPECT_EQ(turbo::uniform(turbo::IntervalClosed, gen, 10, 1000), 11);
  }

  {
    turbo::MockingBitGen gen;
    ON_CALL(turbo::MockUniform<int>(), Call(gen, 1, _))
        .WillByDefault(Return(25));
    ON_CALL(turbo::MockUniform<int>(), Call(gen, Ne(1), _))
        .WillByDefault(Return(99));
    EXPECT_EQ(turbo::uniform(gen, 3, 1000000), 99);
    EXPECT_EQ(turbo::uniform(gen, 1, 1000000), 25);
  }

  {
    turbo::MockingBitGen gen;
    ON_CALL(turbo::MockUniform<int>(), Call(gen, _, _))
        .WillByDefault(Return(145));
    EXPECT_EQ(turbo::uniform(gen, 1, 1000), 145);
    EXPECT_EQ(turbo::uniform(gen, 10, 1000), 145);
    EXPECT_EQ(turbo::uniform(gen, 100, 1000), 145);
  }
}

TEST(AnythingMatcher, WithWillByDefault) {
  using testing::_;
  turbo::MockingBitGen gen;
  std::vector<int> values = {11, 22, 33, 44, 55, 66, 77, 88, 99, 1010};

  ON_CALL(turbo::MockUniform<size_t>(), Call(gen, 0, _))
      .WillByDefault(Return(0));
  for (int i = 0; i < 100; i++) {
    auto& elem = values[turbo::uniform(gen, 0u, values.size())];
    EXPECT_EQ(elem, 11);
  }
}

TEST(BasicMocking, WillByDefaultWithArgs) {
  using testing::_;

  turbo::MockingBitGen gen;
  ON_CALL(turbo::MockPoisson<int>(), Call(gen, _))
      .WillByDefault([](double lambda) {
        return static_cast<int>(std::rint(lambda * 10));
      });
  EXPECT_EQ(turbo::poisson<int>(gen, 1.7), 17);
  EXPECT_EQ(turbo::poisson<int>(gen, 0.03), 0);
}

TEST(MockingBitGen, InSequenceSucceedsInOrder) {
  turbo::MockingBitGen gen;

  testing::InSequence seq;

  EXPECT_CALL(turbo::MockPoisson<int>(), Call(gen, 1.0)).WillOnce(Return(3));
  EXPECT_CALL(turbo::MockPoisson<int>(), Call(gen, 2.0)).WillOnce(Return(4));

  EXPECT_EQ(turbo::poisson<int>(gen, 1.0), 3);
  EXPECT_EQ(turbo::poisson<int>(gen, 2.0), 4);
}

TEST(MockingBitGen, NiceMock) {
  ::testing::NiceMock<turbo::MockingBitGen> gen;
  ON_CALL(turbo::MockUniform<int>(), Call(gen, _, _)).WillByDefault(Return(145));

  ON_CALL(turbo::MockPoisson<int>(), Call(gen, _)).WillByDefault(Return(3));

  EXPECT_EQ(turbo::uniform(gen, 1, 1000), 145);
  EXPECT_EQ(turbo::uniform(gen, 10, 1000), 145);
  EXPECT_EQ(turbo::uniform(gen, 100, 1000), 145);
}

TEST(MockingBitGen, NaggyMock) {
  // This is difficult to test, as only the output matters, so just verify
  // that ON_CALL can be installed. Anything else requires log inspection.
  ::testing::NaggyMock<turbo::MockingBitGen> gen;

  ON_CALL(turbo::MockUniform<int>(), Call(gen, _, _)).WillByDefault(Return(145));
  ON_CALL(turbo::MockPoisson<int>(), Call(gen, _)).WillByDefault(Return(3));

  EXPECT_EQ(turbo::uniform(gen, 1, 1000), 145);
}

TEST(MockingBitGen, StrictMock_NotEnough) {
  EXPECT_NONFATAL_FAILURE(
      []() {
        ::testing::StrictMock<turbo::MockingBitGen> gen;
        EXPECT_CALL(turbo::MockUniform<int>(), Call(gen, _, _))
            .WillOnce(Return(145));
      }(),
      "unsatisfied and active");
}

TEST(MockingBitGen, StrictMock_TooMany) {
  ::testing::StrictMock<turbo::MockingBitGen> gen;

  EXPECT_CALL(turbo::MockUniform<int>(), Call(gen, _, _)).WillOnce(Return(145));
  EXPECT_EQ(turbo::uniform(gen, 1, 1000), 145);

  EXPECT_NONFATAL_FAILURE(
      [&]() { EXPECT_EQ(turbo::uniform(gen, 10, 1000), 0); }(),
      "over-saturated and active");
}

}  // namespace
