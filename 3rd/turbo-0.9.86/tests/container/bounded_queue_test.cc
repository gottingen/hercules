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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "turbo/container/bounded_queue.h"

TEST_CASE("BoundedQueueTest, sanity") {
    const int N = 36;
    char storage[N * sizeof(int)];
    turbo::bounded_queue<int> q(storage, sizeof(storage), turbo::NOT_OWN_STORAGE);
    REQUIRE_EQ(0ul, q.size());
    REQUIRE(q.empty());
    REQUIRE_EQ(nullptr, q.top());
    REQUIRE_EQ(nullptr, q.bottom());
    for (int i = 1; i <= N; ++i) {
        if (i % 2 == 0) {
            REQUIRE(q.push(i));
        } else {
            int *p = q.push();
            REQUIRE(p);
            *p = i;
        }
        REQUIRE_EQ((size_t) i, q.size());
        REQUIRE_EQ(1, *q.top());
        REQUIRE_EQ(i, *q.bottom());
    }
    REQUIRE_FALSE(q.push(N + 1));
    REQUIRE_FALSE(q.push_top(N + 1));
    REQUIRE_EQ((size_t) N, q.size());
    REQUIRE_FALSE(q.empty());
    REQUIRE(q.full());

    for (int i = 1; i <= N; ++i) {
        REQUIRE_EQ(i, *q.top());
        REQUIRE_EQ(N, *q.bottom());
        if (i % 2 == 0) {
            int tmp = 0;
            REQUIRE(q.pop(&tmp));
            REQUIRE_EQ(tmp, i);
        } else {
            REQUIRE(q.pop());
        }
        REQUIRE_EQ((size_t) (N - i), q.size());
    }
    REQUIRE_EQ(0ul, q.size());
    REQUIRE(q.empty());
    REQUIRE_FALSE(q.full());
    REQUIRE_FALSE(q.pop());

    for (int i = 1; i <= N; ++i) {
        if (i % 2 == 0) {
            REQUIRE(q.push_top(i));
        } else {
            int *p = q.push_top();
            REQUIRE(p);
            *p = i;
        }
        REQUIRE_EQ((size_t) i, q.size());
        REQUIRE_EQ(i, *q.top());
        REQUIRE_EQ(1, *q.bottom());
    }
    REQUIRE_FALSE(q.push(N + 1));
    REQUIRE_FALSE(q.push_top(N + 1));
    REQUIRE_EQ((size_t) N, q.size());
    REQUIRE_FALSE(q.empty());
    REQUIRE(q.full());

    for (int i = 1; i <= N; ++i) {
        REQUIRE_EQ(N, *q.top());
        REQUIRE_EQ(i, *q.bottom());
        if (i % 2 == 0) {
            int tmp = 0;
            REQUIRE(q.pop_bottom(&tmp));
            REQUIRE_EQ(tmp, i);
        } else {
            REQUIRE(q.pop_bottom());
        }
        REQUIRE_EQ((size_t) (N - i), q.size());
    }
    REQUIRE_EQ(0ul, q.size());
    REQUIRE(q.empty());
    REQUIRE_FALSE(q.full());
    REQUIRE_FALSE(q.pop());
}
