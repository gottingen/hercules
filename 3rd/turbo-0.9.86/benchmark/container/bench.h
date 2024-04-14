#pragma once

#include <chrono>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>
#include "turbo/container/flat_hash_map.h"
#include "turbo/container/node_hash_map.h"
#include "robin_hood.h"

static const char* MapName = "turbo::node_hash_map";

static const char* HashName = "turbo::Hash";
template <class Key, class Val, class H = turbo::Hash<Key>>
using Map = turbo::node_hash_map<Key, Val, H>;

#include "turbo/memory/memory_info.h"

// inline
#ifdef _WIN32
#define BENCHMARK_NOINLINE __declspec(noinline)
#else
#if __GNUC__ >= 4
#define BENCHMARK_NOINLINE __attribute__((noinline))
#else
#define BENCHMARK_NOINLINE
#endif
#endif

class Bench {
public:
    // use highresolution clock only if it is steady (never adjusted by the system).
    using clock =
        std::conditional<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock, std::chrono::steady_clock>::type;

    Bench(std::string testName)
        : mTestName(std::move(testName)) {
        mInitialPeakRss = turbo::get_peak_memory_used();
    }

    inline void beginMeasure(const char* measurementName) {
        ++mMeasureCount;
        show_tags(measurementName);
        mStartTime = clock::now();
    }

    inline void endMeasure(uint64_t expected_result, uint64_t actual_result) {
        auto const end = clock::now();
        show_result(end, expected_result, actual_result);
    }

private:
    BENCHMARK_NOINLINE void show_tags(const char* measurementName) {

        std::cout << mQuote << MapName << mQuote << mSep;
        std::cout << mQuote << HashName << mQuote << mSep;
        std::cout << mQuote << mTestName << mQuote << mSep;
        std::cout << mQuote << std::setfill('0') << std::setw(2) << mMeasureCount << mQuote << mSep;
        std::cout << mQuote << measurementName << mQuote << mSep;
        std::cout.flush();
    }

    BENCHMARK_NOINLINE void show_result(clock::time_point end, uint64_t expected_result, uint64_t actual_result) {
        auto runtime_sec = std::chrono::duration<double>(end - mStartTime).count();

        auto peakRss = turbo::get_peak_memory_used();
        if (peakRss > mInitialPeakRss) {
            peakRss -= mInitialPeakRss;
        } else {
            peakRss = 0;
        }

        std::cout << actual_result << mSep << runtime_sec << mSep << (peakRss / 1048576.0);
        if (actual_result != expected_result) {
            std::cout << mSep << mQuote << "ERROR: expected " << expected_result << " but got " << actual_result << mQuote;
        }
        std::cout << std::endl;
    }

    clock::time_point mStartTime;
    const char* mSep = "; ";
    const char* mQuote = "\"";
    std::string const mTestName;
    size_t mInitialPeakRss;
    size_t mMeasureCount = 0;
};

class BenchRegistry {
public:
    using Fn = std::function<void(Bench&)>;
    using NameToFn = std::map<std::string, Fn>;

    BenchRegistry(const char* name, Fn fn) {
        if (!nameToFn().emplace(name, fn).second) {
            std::cerr << "benchmark with name '" << name << "' already exists!" << std::endl;
            throw std::exception();
        }
    }

    static void list() {
        for (auto const& nameFn : nameToFn()) {
            std::cout << nameFn.first << std::endl;
        }
    }

    static NameToFn& nameToFn() {
        static NameToFn sNameToFn;
        return sNameToFn;
    }

    static int run(const char* name) {
        auto it = BenchRegistry::nameToFn().find(name);
        if (it == BenchRegistry::nameToFn().end()) {
            return -1;
        }
        try {
            Bench bench(it->first);
            it->second(bench);
        } catch (...) {
            // make sure we get a newline, even in case of an exception.
            std::cout << "\"EXCEPTION\"" << std::endl;
        }
        return 0;
    }
};

#define BENCHMARK_PP_CAT(a, b) BENCHMARK_PP_INTERNAL_CAT(a, b)
#define BENCHMARK_PP_INTERNAL_CAT(a, b) a##b
#define BENCHMARK(f)                                                                                                                                 \
    static void f(Bench& bench);                                                                                                                     \
    static BenchRegistry BENCHMARK_PP_CAT(reg, __LINE__)(#f, f);                                                                                     \
    static void f(Bench& bench)
