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
//
#include "turbo/flags/flags.h"
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

int main(int argc, char **argv) {
    turbo::App app{"An app to practice mixing unlimited arguments, but still recover the original order."};

    std::vector<int> foos;
    auto *foo = app.add_option("--foo,-f", foos, "Some unlimited argument");

    std::vector<int> bars;
    auto *bar = app.add_option("--bar", bars, "Some unlimited argument");

    app.add_flag("--z,--x", "Random other flags");

    // Standard parsing lines (copy and paste in, or use TURBO_FLAGS_PARSE)
    try {
        app.parse(argc, argv);
    } catch(const turbo::ParseError &e) {
        return app.exit(e);
    }

    // I prefer using the back and popping
    std::reverse(std::begin(foos), std::end(foos));
    std::reverse(std::begin(bars), std::end(bars));

    std::vector<std::pair<std::string, int>> keyval;
    for(auto *option : app.parse_order()) {
        if(option == foo) {
            keyval.emplace_back("foo", foos.back());
            foos.pop_back();
        }
        if(option == bar) {
            keyval.emplace_back("bar", bars.back());
            bars.pop_back();
        }
    }

    // Prove the vector is correct
    for(auto &pair : keyval) {
        std::cout << pair.first << " : " << pair.second << std::endl;
    }
}
