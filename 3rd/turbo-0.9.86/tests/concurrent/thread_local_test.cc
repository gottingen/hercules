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
#include "turbo/concurrent/thread_local.h"
#include "turbo/concurrent/write_local.h"
#include "turbo/times/clock.h"
#include "turbo/format/print.h"
#include <thread>
/*
TEST_CASE("thread local") {
    turbo::ThreadLocal<int> tl;
    *tl = 1;
    tl = 2;
    bool stop = false;
    CHECK_EQ(*tl,  2);

    auto t1_func = [&tl, &stop]() {
        *tl = 3;
        while (!stop) {
            std::this_thread::yield();
            CHECK_EQ(*tl, 3);
            turbo::sleep_for(turbo::Duration::milliseconds(1));
        }
    };

    auto t2_func = [&tl, &stop]() {
        *tl = 4;
        while (!stop) {
            std::this_thread::yield();
            CHECK_EQ(*tl, 4);
            turbo::sleep_for(turbo::Duration::milliseconds(1));
        }
    };

    auto t1 = std::thread(t1_func);
    auto t2 = std::thread(t2_func);

    int i = 0;
    while (i < 100) {
        CHECK_EQ(*tl,  2);
        ++i;
        turbo::sleep_for(turbo::Duration::milliseconds(100));
    }
    stop = true;
    t1.join();
    t2.join();
}
*/
TEST_CASE("linkd_thread_local") {
    turbo::WriteLocal<int> tl;
    tl.set(2);
    bool stop = false;
    CHECK_EQ(tl.get(),  2);

    auto t1_func = [&tl, &stop]() {
        tl.set(3);
        while (!stop) {
            std::this_thread::yield();
            CHECK_EQ(tl.get(), 3);
            turbo::sleep_for(turbo::Duration::milliseconds(1));
        }
    };

    auto t2_func = [&tl, &stop]() {
        tl.set(4);
        while (!stop) {
            std::this_thread::yield();
            CHECK_EQ(tl.get(), 4);
            turbo::sleep_for(turbo::Duration::milliseconds(1));
        }
    };

    auto t1 = std::thread(t1_func);
    auto t2 = std::thread(t2_func);
    turbo::sleep_for(turbo::Duration::milliseconds(100));
    std::vector<int> values;
    tl.list(values);
    turbo::println("values:{}", values);
    int i = 0;
    while (i < 20) {
        CHECK_EQ(tl.get(),  2);
        ++i;
        turbo::sleep_for(turbo::Duration::milliseconds(100));
    }
    values.clear();
    tl.list(values);
    turbo::println("values:{}", values);
    stop = true;
    t1.join();
    t2.join();
    turbo::WriteLocal<int> tl1;
    auto common_func = [&tl1, &stop]() {
        tl1.set(4);
        while (!stop) {
            std::this_thread::yield();
            turbo::Println("{}", tl1.get());
            turbo::sleep_for(turbo::Duration::milliseconds(100));
        }
    };

    stop = false;

    auto t3 = std::thread(common_func);
    auto t4 = std::thread(common_func);
    i = 0;
    while (i < 5) {
        turbo::Println("{}", tl1.get());
        ++i;
        turbo::sleep_for(turbo::Duration::milliseconds(100));
    }
    tl1.merge_global(18);
    while (i < 10) {
        turbo::Println("{}", tl1.get());
        ++i;
        turbo::sleep_for(turbo::Duration::milliseconds(100));
    }
    values.clear();
    tl1.list(values);
    turbo::println("values:{}", values);
    stop = true;
    t3.join();
    t4.join();
}
