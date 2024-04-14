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

#include "turbo/strings/ascii.h"

#include <cctype>
#include <string>
#include <array>
#include <random>

#include "benchmark/benchmark.h"

namespace {

std::array<unsigned char, 256> MakeShuffledBytes() {
  std::array<unsigned char, 256> bytes;
  for (size_t i = 0; i < 256; ++i) bytes[i] = static_cast<unsigned char>(i);
  std::random_device rd;
  std::seed_seq seed({rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()});
  std::mt19937 g(seed);
  std::shuffle(bytes.begin(), bytes.end(), g);
  return bytes;
}

template <typename Function>
void AsciiBenchmark(benchmark::State& state, Function f) {
  std::array<unsigned char, 256> bytes = MakeShuffledBytes();
  size_t sum = 0;
  for (auto _ : state) {
    for (unsigned char b : bytes) sum += f(b) ? 1 : 0;
  }
  // Make a copy of `sum` before calling `DoNotOptimize` to make sure that `sum`
  // can be put in a CPU register and not degrade performance in the loop above.
  size_t sum2 = sum;
  benchmark::DoNotOptimize(sum2);
  state.SetBytesProcessed(state.iterations() * bytes.size());
}

using StdAsciiFunction = int (*)(int);
template <StdAsciiFunction f>
void BM_Ascii(benchmark::State& state) {
  AsciiBenchmark(state, f);
}

using TurboAsciiIsFunction = bool (*)(unsigned char);
template <TurboAsciiIsFunction f>
void BM_Ascii(benchmark::State& state) {
  AsciiBenchmark(state, f);
}

using TurboAsciiToFunction = char (*)(unsigned char);
template <TurboAsciiToFunction f>
void BM_Ascii(benchmark::State& state) {
  AsciiBenchmark(state, f);
}

inline char Noop(unsigned char b) { return static_cast<char>(b); }

BENCHMARK_TEMPLATE(BM_Ascii, Noop);
BENCHMARK_TEMPLATE(BM_Ascii, std::isalpha);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_alpha);
BENCHMARK_TEMPLATE(BM_Ascii, std::isdigit);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_digit);
BENCHMARK_TEMPLATE(BM_Ascii, std::isalnum);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_alnum);
BENCHMARK_TEMPLATE(BM_Ascii, std::isspace);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_space);
BENCHMARK_TEMPLATE(BM_Ascii, std::ispunct);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_punct);
BENCHMARK_TEMPLATE(BM_Ascii, std::isblank);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_blank);
BENCHMARK_TEMPLATE(BM_Ascii, std::iscntrl);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_cntrl);
BENCHMARK_TEMPLATE(BM_Ascii, std::isxdigit);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_xdigit);
BENCHMARK_TEMPLATE(BM_Ascii, std::isprint);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_print);
BENCHMARK_TEMPLATE(BM_Ascii, std::isgraph);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_graph);
BENCHMARK_TEMPLATE(BM_Ascii, std::isupper);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_upper);
BENCHMARK_TEMPLATE(BM_Ascii, std::islower);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_lower);
BENCHMARK_TEMPLATE(BM_Ascii, isascii);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_is_ascii);
BENCHMARK_TEMPLATE(BM_Ascii, std::tolower);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_to_lower);
BENCHMARK_TEMPLATE(BM_Ascii, std::toupper);
BENCHMARK_TEMPLATE(BM_Ascii, turbo::ascii_to_upper);

static void BM_StrToLower(benchmark::State& state) {
  const int size = state.range(0);
  std::string s(size, 'X');
  for (auto _ : state) {
    benchmark::DoNotOptimize(turbo::str_to_lower(s));
  }
}
BENCHMARK(BM_StrToLower)->Range(1, 1 << 20);

static void BM_StrToUpper(benchmark::State& state) {
  const int size = state.range(0);
  std::string s(size, 'x');
  for (auto _ : state) {
    benchmark::DoNotOptimize(turbo::str_to_upper(s));
  }
}
BENCHMARK(BM_StrToUpper)->Range(1, 1 << 20);

}  // namespace
