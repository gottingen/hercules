// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
//
// Created by jeff on 24-1-6.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/memory/atomic_object_pool.h"

// --------------------------------------------------------
// Testcase: ObjectPool.Sequential
// --------------------------------------------------------
struct Poolable {
    std::string str;
    std::vector<int> vec;
    int a;
    char b;

    TURBO_ENABLE_POOLABLE_ON_THIS;
};

TEST_CASE("ObjectPool.Sequential" * doctest::timeout(300)) {

    for (unsigned w = 1; w <= 4; w++) {

        turbo::AtomicObjectPool<Poolable> pool(w);

        REQUIRE(pool.num_heaps() > 0);
        REQUIRE(pool.num_local_heaps() > 0);
        REQUIRE(pool.num_global_heaps() > 0);
        REQUIRE(pool.num_bins_per_local_heap() > 0);
        REQUIRE(pool.num_objects_per_bin() > 0);
        REQUIRE(pool.num_objects_per_block() > 0);
        REQUIRE(pool.emptiness_threshold() > 0);

        // fill out all objects
        size_t N = 100 * pool.num_objects_per_block();

        std::set<Poolable *> set;

        for (size_t i = 0; i < N; ++i) {
            auto item = pool.animate();
            REQUIRE(set.find(item) == set.end());
            set.insert(item);
        }

        REQUIRE(set.size() == N);

        for (auto s: set) {
            pool.recycle(s);
        }

        REQUIRE(N == pool.capacity());
        REQUIRE(N == pool.num_available_objects());
        REQUIRE(0 == pool.num_allocated_objects());

        for (size_t i = 0; i < N; ++i) {
            auto item = pool.animate();
            REQUIRE(set.find(item) != set.end());
        }

        REQUIRE(pool.num_available_objects() == 0);
        REQUIRE(pool.num_allocated_objects() == N);
    }
}



// --------------------------------------------------------
// Testcase: ObjectPool.Threaded
// --------------------------------------------------------

template<typename T>
void threaded_objectpool(unsigned W) {

    turbo::AtomicObjectPool<T> pool;

    std::vector<std::thread> threads;

    for (unsigned w = 0; w < W; ++w) {
        threads.emplace_back([&pool]() {
            std::vector<T *> items;
            for (int i = 0; i < 65536; ++i) {
                auto item = pool.animate();
                items.push_back(item);
            }
            for (auto item: items) {
                pool.recycle(item);
            }
        });
    }

    for (auto &thread: threads) {
        thread.join();
    }

    REQUIRE(pool.num_allocated_objects() == 0);
    REQUIRE(pool.num_available_objects() == pool.capacity());
}

TEST_CASE("ObjectPool.1thread" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(1);
}

TEST_CASE("ObjectPool.2threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(2);
}

TEST_CASE("ObjectPool.3threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(3);
}

TEST_CASE("ObjectPool.4threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(4);
}

TEST_CASE("ObjectPool.5threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(5);
}

TEST_CASE("ObjectPool.6threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(6);
}

TEST_CASE("ObjectPool.7threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(7);
}

TEST_CASE("ObjectPool.8threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(8);
}

TEST_CASE("ObjectPool.9threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(9);
}

TEST_CASE("ObjectPool.10threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(10);
}

TEST_CASE("ObjectPool.11threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(11);
}

TEST_CASE("ObjectPool.12threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(12);
}

TEST_CASE("ObjectPool.13threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(13);
}

TEST_CASE("ObjectPool.14threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(14);
}

TEST_CASE("ObjectPool.15threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(15);
}

TEST_CASE("ObjectPool.16threads" * doctest::timeout(300)) {
    threaded_objectpool<Poolable>(16);
}