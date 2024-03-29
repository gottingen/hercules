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

#include <collie/testing/doctest.h>
#include <collie/container/inlined_vector.h>

TEST_CASE("InlinedVector" * doctest::timeout(300)) {

    //SUBCASE("constructor")
    {
        collie::InlinedVector<int> vec1;
        REQUIRE_EQ(vec1.size(), 0);
        REQUIRE_EQ(vec1.empty(), true);

        collie::InlinedVector<int, 4> vec2;
        REQUIRE_NE(vec2.data(), nullptr);
        REQUIRE_EQ(vec2.size(), 0);
        REQUIRE_EQ(vec2.empty(), true);
        REQUIRE_EQ(vec2.capacity(), 4);
    }

    //SUBCASE("constructor_n")
    {
        for (int N = 0; N <= 65536; N = (N ? N << 1 : 1)) {
            collie::InlinedVector<int> vec(N);
            REQUIRE_EQ(vec.size(), N);
            REQUIRE_EQ(vec.empty(), (N == 0));
            REQUIRE_GE(vec.max_size(), vec.size());
            REQUIRE_GE(vec.capacity(), vec.size());
        }
    }

    //SUBCASE("copy_constructor")
    {
        for (int N = 0; N <= 65536; N = (N ? N << 1 : 1)) {
            collie::InlinedVector<int> vec1(N);
            for (auto &item: vec1) {
                item = N;
            }

            collie::InlinedVector<int> vec2(vec1);
            REQUIRE_EQ(vec1.size(), N);
            REQUIRE_EQ(vec2.size(), N);
            for (size_t i = 0; i < vec1.size(); ++i) {
                REQUIRE_EQ(vec1[i], vec2[i]);
                REQUIRE_EQ(vec1[i], N);
            }
        }
    }

    //SUBCASE("move_constructor")
    {
        for (int N = 0; N <= 65536; N = (N ? N << 1 : 1)) {
            collie::InlinedVector<int> vec1(N);
            for (auto &item: vec1) {
                item = N;
            }

            collie::InlinedVector<int> vec2(std::move(vec1));
            REQUIRE_EQ(vec1.size(), 0);
            REQUIRE_EQ(vec1.empty(), true);
            REQUIRE_EQ(vec2.size(), N);

            for (size_t i = 0; i < vec2.size(); ++i) {
                REQUIRE_EQ(vec2[i], N);
            }
        }
    }

    //SUBCASE("push_back")
    {
        for (int N = 0; N <= 65536; N = (N ? N << 1 : 1)) {
            collie::InlinedVector<int> vec;
            size_t pcap{0};
            size_t ncap{0};
            for (int n = 0; n < N; ++n) {
                vec.push_back(n);
                REQUIRE_EQ(vec.size(), n + 1);
                ncap = vec.capacity();
                REQUIRE_GE(ncap, pcap);
                pcap = ncap;
            }
            for (int n = 0; n < N; ++n) {
                REQUIRE_EQ(vec[n], n);
            }
            REQUIRE_EQ(vec.empty(), (N == 0));
        }
    }

    //SUBCASE("pop_back")
    {
        size_t size{0};
        size_t pcap{0};
        size_t ncap{0};
        collie::InlinedVector<int> vec;
        for (int N = 0; N <= 65536; N = (N ? N << 1 : N + 1)) {
            vec.push_back(N);
            ++size;
            REQUIRE_EQ(vec.size(), size);
            if (N % 4 == 0) {
                vec.pop_back();
                --size;
                REQUIRE_EQ(vec.size(), size);
            }
            ncap = vec.capacity();
            REQUIRE_GE(ncap, pcap);
            pcap = ncap;
        }
        REQUIRE_EQ(vec.size(), size);
        for (size_t i = 0; i < vec.size(); ++i) {
            REQUIRE_NE(vec[i] % 4, 0);
        }
    }

    //SUBCASE("iterator")
    {
        for (int N = 0; N <= 65536; N = (N ? N << 1 : 1)) {
            collie::InlinedVector<int> vec;
            for (int n = 0; n < N; ++n) {
                vec.push_back(n);
                REQUIRE_EQ(vec.size(), n + 1);
            }

            // non-constant iterator
            {
                int val{0};
                for (auto item: vec) {
                    REQUIRE_EQ(item, val);
                    ++val;
                }
            }

            // constant iterator
            {
                int val{0};
                for (const auto &item: vec) {
                    REQUIRE_EQ(item, val);
                    ++val;
                }
            }

            // change the value
            {
                for (auto &item: vec) {
                    item = 1234;
                }
                for (auto &item: vec) {
                    REQUIRE_EQ(item, 1234);
                }
            }
        }
    }

    //SUBCASE("clear")
    {
        for (int N = 0; N <= 65536; N = (N ? N << 1 : 1)) {
            collie::InlinedVector<int> vec(N);
            auto cap = vec.capacity();
            REQUIRE_EQ(vec.size(), N);
            vec.clear();
            REQUIRE_EQ(vec.size(), 0);
            REQUIRE_EQ(vec.capacity(), cap);
        }
    }

    //SUBCASE("comparison")
    {
        for (int N = 0; N <= 65536; N = (N ? N << 1 : 1)) {
            collie::InlinedVector<int> vec1;
            for (int i = 0; i < N; ++i) {
                vec1.push_back(i);
            }
            collie::InlinedVector<int> vec2(vec1);
            REQUIRE_EQ(vec1, vec2);
        }
    }
}