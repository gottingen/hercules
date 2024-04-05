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
// Created by jeff on 24-3-29.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <collie/testing/doctest.h>
#include <collie/base/uuid.h>
#include <mutex>
#include <thread>
#include <algorithm>

void test_threaded_uuid(size_t N) {

    std::vector<collie::UUID> uuids(65536);

    // threaded
    std::mutex mutex;
    std::vector<std::thread> threads;

    for(size_t i=0; i<N; ++i) {
        threads.emplace_back([&](){
            for(int j=0; j<1000; ++j) {
                std::lock_guard<std::mutex> lock(mutex);
                uuids.push_back(collie::UUID());
            }
        });
    }

    for(auto& t : threads) {
        t.join();
    }

    auto size = uuids.size();
    std::sort(uuids.begin(), uuids.end());
    std::unique(uuids.begin(), uuids.end());
    REQUIRE(uuids.size() == size);
}

TEST_CASE("uuid") {

collie::UUID u1, u2, u3, u4;

// Comparator.
REQUIRE(u1 == u1);

// Copy
u2 = u1;
REQUIRE(u1 == u2);

// Move
u3 = std::move(u1);
REQUIRE(u2 == u3);

// Copy constructor
collie::UUID u5(u4);
REQUIRE(u5 == u4);

// Move constructor.
collie::UUID u6(std::move(u4));
REQUIRE(u5 == u6);

// Uniqueness
std::vector<collie::UUID> uuids(65536);
std::sort(uuids.begin(), uuids.end());
std::unique(uuids.begin(), uuids.end());
REQUIRE(uuids.size() == 65536);

}

TEST_CASE("uuid.10threads") {
test_threaded_uuid(10);
}

TEST_CASE("uuid.100threads") {
test_threaded_uuid(100);
}