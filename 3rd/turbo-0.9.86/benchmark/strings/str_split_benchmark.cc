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

#include "turbo/strings/str_split.h"

#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "benchmark/benchmark.h"
#include "turbo/base/internal/raw_logging.h"
#include "turbo/strings/string_view.h"

namespace {

std::string MakeTestString(int desired_length) {
  static const int kAverageValueLen = 25;
  std::string test(desired_length * kAverageValueLen, 'x');
  for (int i = 1; i < test.size(); i += kAverageValueLen) {
    test[i] = ';';
  }
  return test;
}

void BM_Split2StringView(benchmark::State& state) {
  std::string test = MakeTestString(state.range(0));
  for (auto _ : state) {
    std::vector<std::string_view> result = turbo::str_split(test, ';');
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK_RANGE(BM_Split2StringView, 0, 1 << 20);

static const std::string_view kDelimiters = ";:,.";

std::string MakeMultiDelimiterTestString(int desired_length) {
  static const int kAverageValueLen = 25;
  std::string test(desired_length * kAverageValueLen, 'x');
  for (int i = 0; i * kAverageValueLen < test.size(); ++i) {
    // Cycle through a variety of delimiters.
    test[i * kAverageValueLen] = kDelimiters[i % kDelimiters.size()];
  }
  return test;
}

// Measure str_split with by_any_char with four delimiters to choose from.
void BM_Split2StringViewByAnyChar(benchmark::State& state) {
  std::string test = MakeMultiDelimiterTestString(state.range(0));
  for (auto _ : state) {
    std::vector<std::string_view> result =
        turbo::str_split(test, turbo::by_any_char(kDelimiters));
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK_RANGE(BM_Split2StringViewByAnyChar, 0, 1 << 20);

void BM_Split2StringViewLifted(benchmark::State& state) {
  std::string test = MakeTestString(state.range(0));
  std::vector<std::string_view> result;
  for (auto _ : state) {
    result = turbo::str_split(test, ';');
  }
  benchmark::DoNotOptimize(result);
}
BENCHMARK_RANGE(BM_Split2StringViewLifted, 0, 1 << 20);

void BM_Split2String(benchmark::State& state) {
  std::string test = MakeTestString(state.range(0));
  for (auto _ : state) {
    std::vector<std::string> result = turbo::str_split(test, ';');
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK_RANGE(BM_Split2String, 0, 1 << 20);

// This benchmark is for comparing Split2 to Split1 (SplitStringUsing). In
// particular, this benchmark uses skip_empty() to match SplitStringUsing's
// behavior.
void BM_Split2SplitStringUsing(benchmark::State& state) {
  std::string test = MakeTestString(state.range(0));
  for (auto _ : state) {
    std::vector<std::string> result =
        turbo::str_split(test, ';', turbo::skip_empty());
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK_RANGE(BM_Split2SplitStringUsing, 0, 1 << 20);

void BM_SplitStringToUnorderedSet(benchmark::State& state) {
  const int len = state.range(0);
  std::string test(len, 'x');
  for (int i = 1; i < len; i += 2) {
    test[i] = ';';
  }
  for (auto _ : state) {
    std::unordered_set<std::string> result =
        turbo::str_split(test, ':', turbo::skip_empty());
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK_RANGE(BM_SplitStringToUnorderedSet, 0, 1 << 20);

void BM_SplitStringToUnorderedMap(benchmark::State& state) {
  const int len = state.range(0);
  std::string test(len, 'x');
  for (int i = 1; i < len; i += 2) {
    test[i] = ';';
  }
  for (auto _ : state) {
    std::unordered_map<std::string, std::string> result =
        turbo::str_split(test, ':', turbo::skip_empty());
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK_RANGE(BM_SplitStringToUnorderedMap, 0, 1 << 20);

void BM_SplitStringAllowEmpty(benchmark::State& state) {
  const int len = state.range(0);
  std::string test(len, 'x');
  for (int i = 1; i < len; i += 2) {
    test[i] = ';';
  }
  for (auto _ : state) {
    std::vector<std::string> result = turbo::str_split(test, ';');
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK_RANGE(BM_SplitStringAllowEmpty, 0, 1 << 20);

struct OneCharLiteral {
  char operator()() const { return 'X'; }
};

struct OneCharStringLiteral {
  const char* operator()() const { return "X"; }
};

template <typename DelimiterFactory>
void BM_SplitStringWithOneChar(benchmark::State& state) {
  const auto delimiter = DelimiterFactory()();
  std::vector<std::string_view> pieces;
  size_t v = 0;
  for (auto _ : state) {
    pieces = turbo::str_split("The quick brown fox jumps over the lazy dog",
                            delimiter);
    v += pieces.size();
  }
  TURBO_RAW_CHECK(v == state.iterations(), "");
}
BENCHMARK_TEMPLATE(BM_SplitStringWithOneChar, OneCharLiteral);
BENCHMARK_TEMPLATE(BM_SplitStringWithOneChar, OneCharStringLiteral);

template <typename DelimiterFactory>
void BM_SplitStringWithOneCharNoVector(benchmark::State& state) {
  const auto delimiter = DelimiterFactory()();
  size_t v = 0;
  for (auto _ : state) {
    auto splitter = turbo::str_split(
        "The quick brown fox jumps over the lazy dog", delimiter);
    v += std::distance(splitter.begin(), splitter.end());
  }
  TURBO_RAW_CHECK(v == state.iterations(), "");
}
BENCHMARK_TEMPLATE(BM_SplitStringWithOneCharNoVector, OneCharLiteral);
BENCHMARK_TEMPLATE(BM_SplitStringWithOneCharNoVector, OneCharStringLiteral);

}  // namespace
