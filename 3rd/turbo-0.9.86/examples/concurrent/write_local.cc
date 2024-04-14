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
// Created by jeff on 23-12-17.
//

#include "turbo/concurrent/thread_local.h"
#include "turbo/concurrent/write_local.h"
#include "turbo/times/clock.h"
#include "turbo/format/print.h"
#include <thread>
#include <tuple>
using namespace turbo;

int main() {

    bool stop = false;
    std::vector<int> values;
    turbo::WriteLocal<int> tl1;
    auto common_func = [&tl1, &stop]() {
        tl1.set(4);
        while (!stop) {
            std::this_thread::yield();
            turbo::Println("{}", tl1.get());
            turbo::sleep_for(turbo::Duration::milliseconds(5));
        }
    };

    stop = false;

    auto t3 = std::thread(common_func);
    auto t4 = std::thread(common_func);
    int i = 0;
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