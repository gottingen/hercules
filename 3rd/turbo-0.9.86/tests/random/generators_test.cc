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

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "turbo/random/fwd.h"
#include "turbo/random/random.h"

namespace {

template <typename URBG>
void Testuniform(URBG* gen) {
  // [a, b) default-semantics, inferred types.
  turbo::uniform(*gen, 0, 100);     // int
  turbo::uniform(*gen, 0, 1.0);     // Promoted to double
  turbo::uniform(*gen, 0.0f, 1.0);  // Promoted to double
  turbo::uniform(*gen, 0.0, 1.0);   // double
  turbo::uniform(*gen, -1, 1L);     // Promoted to long

  // Roll a die.
  turbo::uniform(turbo::IntervalClosedClosed, *gen, 1, 6);

  // Get a fraction.
  turbo::uniform(turbo::IntervalOpenOpen, *gen, 0.0, 1.0);

  // Assign a value to a random element.
  std::vector<int> elems = {10, 20, 30, 40, 50};
  elems[turbo::uniform(*gen, 0u, elems.size())] = 5;
  elems[turbo::uniform<size_t>(*gen, 0, elems.size())] = 3;

  // Choose some epsilon around zero.
  turbo::uniform(turbo::IntervalOpenOpen, *gen, -1.0, 1.0);

  // (a, b) semantics, inferred types.
  turbo::uniform(turbo::IntervalOpenOpen, *gen, 0, 1.0);  // Promoted to double

  // Explict overriding of types.
  turbo::uniform<int>(*gen, 0, 100);
  turbo::uniform<int8_t>(*gen, 0, 100);
  turbo::uniform<int16_t>(*gen, 0, 100);
  turbo::uniform<uint16_t>(*gen, 0, 100);
  turbo::uniform<int32_t>(*gen, 0, 1 << 10);
  turbo::uniform<uint32_t>(*gen, 0, 1 << 10);
  turbo::uniform<int64_t>(*gen, 0, 1 << 10);
  turbo::uniform<uint64_t>(*gen, 0, 1 << 10);

  turbo::uniform<float>(*gen, 0.0, 1.0);
  turbo::uniform<float>(*gen, 0, 1);
  turbo::uniform<float>(*gen, -1, 1);
  turbo::uniform<double>(*gen, 0.0, 1.0);

  turbo::uniform<float>(*gen, -1.0, 0);
  turbo::uniform<double>(*gen, -1.0, 0);

  // Tagged
  turbo::uniform<double>(turbo::IntervalClosedClosed, *gen, 0, 1);
  turbo::uniform<double>(turbo::IntervalClosedOpen, *gen, 0, 1);
  turbo::uniform<double>(turbo::IntervalOpenOpen, *gen, 0, 1);
  turbo::uniform<double>(turbo::IntervalOpenClosed, *gen, 0, 1);
  turbo::uniform<double>(turbo::IntervalClosedClosed, *gen, 0, 1);
  turbo::uniform<double>(turbo::IntervalOpenOpen, *gen, 0, 1);

  turbo::uniform<int>(turbo::IntervalClosedClosed, *gen, 0, 100);
  turbo::uniform<int>(turbo::IntervalClosedOpen, *gen, 0, 100);
  turbo::uniform<int>(turbo::IntervalOpenOpen, *gen, 0, 100);
  turbo::uniform<int>(turbo::IntervalOpenClosed, *gen, 0, 100);
  turbo::uniform<int>(turbo::IntervalClosedClosed, *gen, 0, 100);
  turbo::uniform<int>(turbo::IntervalOpenOpen, *gen, 0, 100);

  // With *generator as an R-value reference.
  turbo::uniform<int>(URBG(), 0, 100);
  turbo::uniform<double>(URBG(), 0.0, 1.0);
}

template <typename URBG>
void TestExponential(URBG* gen) {
  turbo::exponential<float>(*gen);
  turbo::exponential<double>(*gen);
  turbo::exponential<double>(URBG());
}

template <typename URBG>
void TestPoisson(URBG* gen) {
  // [rand.dist.pois] Indicates that the std::poisson_distribution
  // is parameterized by IntType, however MSVC does not allow 8-bit
  // types.
  turbo::poisson<int>(*gen);
  turbo::poisson<int16_t>(*gen);
  turbo::poisson<uint16_t>(*gen);
  turbo::poisson<int32_t>(*gen);
  turbo::poisson<uint32_t>(*gen);
  turbo::poisson<int64_t>(*gen);
  turbo::poisson<uint64_t>(*gen);
  turbo::poisson<uint64_t>(URBG());
  turbo::poisson<turbo::int128>(*gen);
  turbo::poisson<turbo::uint128>(*gen);
}

template <typename URBG>
void TestBernoulli(URBG* gen) {
  turbo::bernoulli(*gen, 0.5);
  turbo::bernoulli(*gen, 0.5);
}

template <typename URBG>
void TestZipf(URBG* gen) {
  turbo::zipf<int>(*gen, 100);
  turbo::zipf<int8_t>(*gen, 100);
  turbo::zipf<int16_t>(*gen, 100);
  turbo::zipf<uint16_t>(*gen, 100);
  turbo::zipf<int32_t>(*gen, 1 << 10);
  turbo::zipf<uint32_t>(*gen, 1 << 10);
  turbo::zipf<int64_t>(*gen, 1 << 10);
  turbo::zipf<uint64_t>(*gen, 1 << 10);
  turbo::zipf<uint64_t>(URBG(), 1 << 10);
  turbo::zipf<turbo::int128>(*gen, 1 << 10);
  turbo::zipf<turbo::uint128>(*gen, 1 << 10);
}

template <typename URBG>
void TestGaussian(URBG* gen) {
  turbo::gaussian<float>(*gen, 1.0, 1.0);
  turbo::gaussian<double>(*gen, 1.0, 1.0);
  turbo::gaussian<double>(URBG(), 1.0, 1.0);
}

template <typename URBG>
void TestLogNormal(URBG* gen) {
  turbo::log_uniform<int>(*gen, 0, 100);
  turbo::log_uniform<int8_t>(*gen, 0, 100);
  turbo::log_uniform<int16_t>(*gen, 0, 100);
  turbo::log_uniform<uint16_t>(*gen, 0, 100);
  turbo::log_uniform<int32_t>(*gen, 0, 1 << 10);
  turbo::log_uniform<uint32_t>(*gen, 0, 1 << 10);
  turbo::log_uniform<int64_t>(*gen, 0, 1 << 10);
  turbo::log_uniform<uint64_t>(*gen, 0, 1 << 10);
  turbo::log_uniform<uint64_t>(URBG(), 0, 1 << 10);
  turbo::log_uniform<turbo::int128>(*gen, 0, 1 << 10);
  turbo::log_uniform<turbo::uint128>(*gen, 0, 1 << 10);
}

template <typename URBG>
void CompatibilityTest() {
  URBG gen;

  Testuniform(&gen);
  TestExponential(&gen);
  TestPoisson(&gen);
  TestBernoulli(&gen);
  TestZipf(&gen);
  TestGaussian(&gen);
  TestLogNormal(&gen);
}

TEST(std_mt19937_64, Compatibility) {
  // Validate with std::mt19937_64
  CompatibilityTest<std::mt19937_64>();
}

TEST(BitGen, Compatibility) {
  // Validate with turbo::BitGen
  CompatibilityTest<turbo::BitGen>();
}

TEST(InsecureBitGen, Compatibility) {
  // Validate with turbo::InsecureBitGen
  CompatibilityTest<turbo::InsecureBitGen>();
}

}  // namespace
