// Copyright 2023 The Turbo Authors.
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
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include "turbo/concurrent/latch.h"
#include "turbo/times/clock.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"

namespace turbo {

    std::atomic<bool> exiting{false};

    void RunTest() {
        std::size_t local_count = 0, remote_count = 0;
        while (!exiting) {
            auto called = std::make_shared<std::atomic<bool>>(false);
            std::this_thread::yield();  // Wait for thread pool to start.
            Latch l(1);
            auto t = std::thread([&] {
                if (!called->exchange(true)) {
                    std::this_thread::yield();  // Something costly.
                    l.CountDown();
                    ++remote_count;
                }
            });
            std::this_thread::yield();  // Something costly.
            if (!called->exchange(true)) {
                l.CountDown();
                ++local_count;
            }
            l.Wait();
            t.join();
        }
        std::cout << local_count << " " << remote_count << std::endl;
    }

    TEST_CASE("Latch") {
        SUBCASE("Torture") {
            std::thread ts[10];
            for (auto &&t: ts) {
                t = std::thread(RunTest);
            }
            std::this_thread::sleep_for(std::chrono::seconds(10));
            exiting = true;
            for (auto &&t: ts) {
                t.join();
            }
        }

        SUBCASE("CountDownTwo")
        {
            Latch l(2);
            l.ArriveAndWait(2);
            REQUIRE(1);
        }

        SUBCASE("WaitFor")
        {
            Latch l(1);
            REQUIRE_FALSE(l.WaitFor(turbo::Duration::milliseconds(100)));
            l.CountDown();
            REQUIRE(l.WaitFor(turbo::Duration::zero()));
        }

        SUBCASE("WaitUntil")
        {
            Latch l(1);
            REQUIRE_FALSE(l.WaitUntil(turbo::time_now() + turbo::Duration::milliseconds(100)));
            l.CountDown();
            REQUIRE(l.WaitUntil(turbo::time_now()));
        }
    }
}  // namespace turbo