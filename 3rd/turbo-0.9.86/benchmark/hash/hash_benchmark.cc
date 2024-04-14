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

#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "turbo/container/flat_hash_set.h"
#include "turbo/hash/hash.h"
#include "turbo/platform/port.h"
#include "turbo/random/random.h"
#include "turbo/strings/cord.h"
#include "turbo/strings/string_view.h"

namespace {

    using turbo::Hash;

    template<template<typename> class H, typename T>
    void RunBenchmark(benchmark::State &state, T value) {
        H<T> h;
        for (auto _: state) {
            benchmark::DoNotOptimize(value);
            benchmark::DoNotOptimize(h(value));
        }
    }

}  // namespace

template<typename T>
using TurboHash = turbo::Hash<T>;

class TypeErasedInterface {
public:
    virtual ~TypeErasedInterface() = default;

    template<typename H>
    friend H hash_value(H state, const TypeErasedInterface &wrapper) {
        state = H::combine(std::move(state), std::type_index(typeid(wrapper)));
        wrapper.HashValue(turbo::HashState::Create(&state));
        return state;
    }

private:
    virtual void HashValue(turbo::HashState state) const = 0;
};

template<typename T>
struct TypeErasedTurboHash {
    class Wrapper : public TypeErasedInterface {
    public:
        explicit Wrapper(const T &value) : value_(value) {}

    private:
        void HashValue(turbo::HashState state) const override {
            turbo::HashState::combine(std::move(state), value_);
        }

        const T &value_;
    };

    size_t operator()(const T &value) {
        return turbo::Hash<Wrapper>{}(Wrapper(value));
    }
};

template<typename FuncType>
inline FuncType *ODRUseFunction(FuncType *ptr) {
    volatile FuncType *dummy = ptr;
    return dummy;
}

turbo::Cord FlatCord(size_t size) {
    turbo::Cord result(std::string(size, 'a'));
    result.flatten();
    return result;
}

/*
turbo::Cord FragmentedCord(size_t size) {
  const size_t orig_size = size;
  std::vector<std::string> chunks;
  size_t chunk_size = std::max<size_t>(1, size / 10);
  while (size > chunk_size) {
    chunks.push_back(std::string(chunk_size, 'a'));
    size -= chunk_size;
  }
  if (size > 0) {
    chunks.push_back(std::string(size, 'a'));
  }
  turbo::Cord result = turbo::make_fragmented_cord(chunks);
  (void) orig_size;
  assert(result.size() == orig_size);
  return result;
}*/

template<typename T>
std::vector<T> Vector(size_t count) {
    std::vector<T> result;
    for (size_t v = 0; v < count; ++v) {
        result.push_back(v);
    }
    return result;
}

// Bogus type that replicates an unorderd_set's bit mixing, but with
// vector-speed iteration. This is intended to measure the overhead of unordered
// hashing without counting the speed of unordered_set iteration.
template<typename T>
struct FastUnorderedSet {
    explicit FastUnorderedSet(size_t count) {
        for (size_t v = 0; v < count; ++v) {
            values.push_back(v);
        }
    }

    std::vector<T> values;

    template<typename H>
    friend H hash_value(H h, const FastUnorderedSet &fus) {
        return H::combine(H::combine_unordered(std::move(h), fus.values.begin(),
                                               fus.values.end()),
                          fus.values.size());
    }
};

template<typename T>
turbo::flat_hash_set<T> FlatHashSet(size_t count) {
    turbo::flat_hash_set<T> result;
    for (size_t v = 0; v < count; ++v) {
        result.insert(v);
    }
    return result;
}

// Generates a benchmark and a codegen method for the provided types.  The
// codegen method provides a well known entrypoint for dumping assembly.
#define MAKE_BENCHMARK(hash, name, ...)                          \
  namespace {                                                    \
  void BM_##hash##_##name(benchmark::State& state) {             \
    RunBenchmark<hash>(state, __VA_ARGS__);                      \
  }                                                              \
  BENCHMARK(BM_##hash##_##name);                                 \
  }                                                              \
  size_t Codegen##hash##name(const decltype(__VA_ARGS__)& arg);  \
  size_t Codegen##hash##name(const decltype(__VA_ARGS__)& arg) { \
    return hash<decltype(__VA_ARGS__)>{}(arg);                   \
  }                                                              \
  bool turbo_hash_test_odr_use##hash##name =                      \
      ODRUseFunction(&Codegen##hash##name);

MAKE_BENCHMARK(TurboHash, Int32, int32_t{});
MAKE_BENCHMARK(TurboHash, Int64, int64_t{});
MAKE_BENCHMARK(TurboHash, Double, 1.2);
MAKE_BENCHMARK(TurboHash, DoubleZero, 0.0);
MAKE_BENCHMARK(TurboHash, PairInt32Int32, std::pair<int32_t, int32_t>{});
MAKE_BENCHMARK(TurboHash, PairInt64Int64, std::pair<int64_t, int64_t>{});
MAKE_BENCHMARK(TurboHash, TupleInt32BoolInt64,
               std::tuple<int32_t, bool, int64_t>{});
MAKE_BENCHMARK(TurboHash, String_0, std::string());
MAKE_BENCHMARK(TurboHash, String_10, std::string(10, 'a'));
MAKE_BENCHMARK(TurboHash, String_30, std::string(30, 'a'));
MAKE_BENCHMARK(TurboHash, String_90, std::string(90, 'a'));
MAKE_BENCHMARK(TurboHash, String_200, std::string(200, 'a'));
MAKE_BENCHMARK(TurboHash, String_5000, std::string(5000, 'a'));
MAKE_BENCHMARK(TurboHash, Cord_Flat_0, turbo::Cord());
MAKE_BENCHMARK(TurboHash, Cord_Flat_10, FlatCord(10));
MAKE_BENCHMARK(TurboHash, Cord_Flat_30, FlatCord(30));
MAKE_BENCHMARK(TurboHash, Cord_Flat_90, FlatCord(90));
MAKE_BENCHMARK(TurboHash, Cord_Flat_200, FlatCord(200));
MAKE_BENCHMARK(TurboHash, Cord_Flat_5000, FlatCord(5000));
//MAKE_BENCHMARK(TurboHash, Cord_Fragmented_200, FragmentedCord(200));
//MAKE_BENCHMARK(TurboHash, Cord_Fragmented_5000, FragmentedCord(5000));
MAKE_BENCHMARK(TurboHash, VectorInt64_10, Vector<int64_t>(10));
MAKE_BENCHMARK(TurboHash, VectorInt64_100, Vector<int64_t>(100));
MAKE_BENCHMARK(TurboHash, VectorInt64_1000, Vector<int64_t>(1000));
MAKE_BENCHMARK(TurboHash, VectorDouble_10, Vector<double>(10));
MAKE_BENCHMARK(TurboHash, VectorDouble_100, Vector<double>(100));
MAKE_BENCHMARK(TurboHash, VectorDouble_1000, Vector<double>(1000));
MAKE_BENCHMARK(TurboHash, FlatHashSetInt64_10, FlatHashSet<int64_t>(10));
MAKE_BENCHMARK(TurboHash, FlatHashSetInt64_100, FlatHashSet<int64_t>(100));
MAKE_BENCHMARK(TurboHash, FlatHashSetInt64_1000, FlatHashSet<int64_t>(1000));
MAKE_BENCHMARK(TurboHash, FlatHashSetDouble_10, FlatHashSet<double>(10));
MAKE_BENCHMARK(TurboHash, FlatHashSetDouble_100, FlatHashSet<double>(100));
MAKE_BENCHMARK(TurboHash, FlatHashSetDouble_1000, FlatHashSet<double>(1000));
MAKE_BENCHMARK(TurboHash, FastUnorderedSetInt64_1000,
               FastUnorderedSet<int64_t>(1000));
MAKE_BENCHMARK(TurboHash, FastUnorderedSetDouble_1000,
               FastUnorderedSet<double>(1000));
MAKE_BENCHMARK(TurboHash, PairStringString_0,
               std::make_pair(std::string(), std::string()));
MAKE_BENCHMARK(TurboHash, PairStringString_10,
               std::make_pair(std::string(10, 'a'), std::string(10, 'b')));
MAKE_BENCHMARK(TurboHash, PairStringString_30,
               std::make_pair(std::string(30, 'a'), std::string(30, 'b')));
MAKE_BENCHMARK(TurboHash, PairStringString_90,
               std::make_pair(std::string(90, 'a'), std::string(90, 'b')));
MAKE_BENCHMARK(TurboHash, PairStringString_200,
               std::make_pair(std::string(200, 'a'), std::string(200, 'b')));
MAKE_BENCHMARK(TurboHash, PairStringString_5000,
               std::make_pair(std::string(5000, 'a'), std::string(5000, 'b')));

MAKE_BENCHMARK(TypeErasedTurboHash, Int32, int32_t{});
MAKE_BENCHMARK(TypeErasedTurboHash, Int64, int64_t{});
MAKE_BENCHMARK(TypeErasedTurboHash, PairInt32Int32,
               std::pair<int32_t, int32_t>{});
MAKE_BENCHMARK(TypeErasedTurboHash, PairInt64Int64,
               std::pair<int64_t, int64_t>{});
MAKE_BENCHMARK(TypeErasedTurboHash, TupleInt32BoolInt64,
               std::tuple<int32_t, bool, int64_t>{});
MAKE_BENCHMARK(TypeErasedTurboHash, String_0, std::string());
MAKE_BENCHMARK(TypeErasedTurboHash, String_10, std::string(10, 'a'));
MAKE_BENCHMARK(TypeErasedTurboHash, String_30, std::string(30, 'a'));
MAKE_BENCHMARK(TypeErasedTurboHash, String_90, std::string(90, 'a'));
MAKE_BENCHMARK(TypeErasedTurboHash, String_200, std::string(200, 'a'));
MAKE_BENCHMARK(TypeErasedTurboHash, String_5000, std::string(5000, 'a'));
MAKE_BENCHMARK(TypeErasedTurboHash, VectorDouble_10,
               std::vector<double>(10, 1.1));
MAKE_BENCHMARK(TypeErasedTurboHash, VectorDouble_100,
               std::vector<double>(100, 1.1));
MAKE_BENCHMARK(TypeErasedTurboHash, VectorDouble_1000,
               std::vector<double>(1000, 1.1));
MAKE_BENCHMARK(TypeErasedTurboHash, FlatHashSetInt64_10,
               FlatHashSet<int64_t>(10));
MAKE_BENCHMARK(TypeErasedTurboHash, FlatHashSetInt64_100,
               FlatHashSet<int64_t>(100));
MAKE_BENCHMARK(TypeErasedTurboHash, FlatHashSetInt64_1000,
               FlatHashSet<int64_t>(1000));
MAKE_BENCHMARK(TypeErasedTurboHash, FlatHashSetDouble_10,
               FlatHashSet<double>(10));
MAKE_BENCHMARK(TypeErasedTurboHash, FlatHashSetDouble_100,
               FlatHashSet<double>(100));
MAKE_BENCHMARK(TypeErasedTurboHash, FlatHashSetDouble_1000,
               FlatHashSet<double>(1000));
MAKE_BENCHMARK(TypeErasedTurboHash, FastUnorderedSetInt64_1000,
               FastUnorderedSet<int64_t>(1000));
MAKE_BENCHMARK(TypeErasedTurboHash, FastUnorderedSetDouble_1000,
               FastUnorderedSet<double>(1000));

// The latency benchmark attempts to model the speed of the hash function in
// production. When a hash function is used for hashtable lookups it is rarely
// used to hash N items in a tight loop nor on constant sized strings. Instead,
// after hashing there is a potential equality test plus a (usually) large
// amount of user code. To simulate this effectively we introduce a data
// dependency between elements we hash by using the hash of the Nth element as
// the selector of the N+1th element to hash. This isolates the hash function
// code much like in production. As a bonus we use the hash to generate strings
// of size [1,N] (instead of fixed N) to disable perfect branch predictions in
// hash function implementations.
namespace {
// 16kb fits in L1 cache of most CPUs we care about. Keeping memory latency low
// will allow us to attribute most time to CPU which means more accurate
// measurements.
    static constexpr size_t kEntropySize = 16 << 10;
    static char entropy[kEntropySize + 1024];
    TURBO_MAYBE_UNUSED static const bool kInitialized = [] {
        turbo::BitGen gen;
        static_assert(sizeof(entropy) % sizeof(uint64_t) == 0, "");
        for (int i = 0; i != sizeof(entropy); i += sizeof(uint64_t)) {
            auto rand = turbo::uniform<uint64_t>(gen);
            memcpy(&entropy[i], &rand, sizeof(uint64_t));
        }
        return true;
    }();
}  // namespace

template<class T>
struct PodRand {
    static_assert(std::is_pod<T>::value, "");
    static_assert(kEntropySize + sizeof(T) < sizeof(entropy), "");

    T Get(size_t i) const {
        T v;
        memcpy(&v, &entropy[i % kEntropySize], sizeof(T));
        return v;
    }
};

template<size_t N>
struct StringRand {
    static_assert(kEntropySize + N < sizeof(entropy), "");

    std::string_view Get(size_t i) const {
        // This has a small bias towards small numbers. Because max N is ~200 this
        // is very small and prefer to be very fast instead of absolutely accurate.
        // Also we pass N = 2^K+1 so that mod reduces to a bitand.
        size_t s = (i % (N - 1)) + 1;
        return {&entropy[i % kEntropySize], s};
    }
};

#define MAKE_LATENCY_BENCHMARK(hash, name, ...)              \
  namespace {                                                \
  void BM_latency_##hash##_##name(benchmark::State& state) { \
    __VA_ARGS__ r;                                           \
    hash<decltype(r.Get(0))> h;                              \
    size_t i = 871401241;                                    \
    for (auto _ : state) {                                   \
      benchmark::DoNotOptimize(i = h(r.Get(i)));             \
    }                                                        \
  }                                                          \
  BENCHMARK(BM_latency_##hash##_##name);                     \
  }  // namespace

MAKE_LATENCY_BENCHMARK(TurboHash, Int32, PodRand<int32_t>);
MAKE_LATENCY_BENCHMARK(TurboHash, Int64, PodRand<int64_t>);
MAKE_LATENCY_BENCHMARK(TurboHash, String9, StringRand<9>);
MAKE_LATENCY_BENCHMARK(TurboHash, String33, StringRand<33>);
MAKE_LATENCY_BENCHMARK(TurboHash, String65, StringRand<65>);
MAKE_LATENCY_BENCHMARK(TurboHash, String257, StringRand<257>);
